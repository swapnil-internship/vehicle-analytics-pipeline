import os
import queue
import threading
from getpass import getpass

import mysql.connector
from mysql.connector import pooling


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "vehicle_analytics"),
    "connection_timeout": 5,
}

# Events wait here until the database worker processes them.
_event_queue = queue.Queue(maxsize=1000)

_pool = None
_worker = None
_worker_lock = threading.Lock()
_database_errors = 0


def _ensure_run_id_schema(connection):
    """
    Add run_id to an existing vehicle_events table if it is not present.
    Existing records are preserved and receive NULL for run_id.
    """
    cursor = connection.cursor()
    try:
        cursor.execute("SHOW COLUMNS FROM vehicle_events LIKE 'run_id'")
        if cursor.fetchone() is None:
            cursor.execute(
                "ALTER TABLE vehicle_events "
                "ADD COLUMN run_id CHAR(36) NULL"
            )
            connection.commit()
            print("Added run_id column to vehicle_events.")

        cursor.execute("SHOW INDEX FROM vehicle_events WHERE Key_name = 'idx_vehicle_events_run_id'")
        if cursor.fetchone() is None:
            cursor.execute(
                "CREATE INDEX idx_vehicle_events_run_id "
                "ON vehicle_events (run_id)"
            )
            connection.commit()
            print("Added index for vehicle_events.run_id.")
    finally:
        cursor.close()


def initialize_database():
    """Connect to MySQL, ensure run_id schema, and start the background writer."""
    global _pool, _worker

    with _worker_lock:
        if _worker is not None and _worker.is_alive():
            return

        config = DB_CONFIG.copy()

        if not config["password"]:
            config["password"] = getpass("Enter your MySQL password: ")

        _pool = pooling.MySQLConnectionPool(
            pool_name="vehicle_pool",
            pool_size=3,
            **config,
        )

        # Verify the connection and apply a backward-compatible schema migration.
        connection = _pool.get_connection()
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE()")
            database_name = cursor.fetchone()[0]
            cursor.close()

            _ensure_run_id_schema(connection)
            print(f"Connected to MySQL database: {database_name}")
        finally:
            connection.close()

        _worker = threading.Thread(
            target=_database_worker,
            name="MySQL-Event-Writer",
            daemon=True,
        )
        _worker.start()
        print("Asynchronous database writer started.")


def _insert_event(event):
    """Perform the actual database insert in the worker thread."""
    connection = _pool.get_connection()
    cursor = None

    try:
        cursor = connection.cursor()
        query = """
            INSERT INTO vehicle_events
            (
                run_id,
                track_id,
                vehicle_type,
                vehicle_color,
                event_timestamp,
                direction
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        cursor.execute(
            query,
            (
                event.get("run_id"),
                int(event["track_id"]),
                event["vehicle_type"],
                event["vehicle_color"],
                float(event["timestamp_seconds"]),
                event["direction"],
            ),
        )
        connection.commit()
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def _database_worker():
    """Consume queued events and persist them in the background."""
    global _database_errors

    while True:
        event = _event_queue.get()
        try:
            if event is None:
                return

            try:
                _insert_event(event)
            except Exception as error:
                _database_errors += 1
                print(
                    "DATABASE WRITE FAILED: "
                    f"run_id={event.get('run_id')}, "
                    f"track_id={event['track_id']}: {error}"
                )
        finally:
            _event_queue.task_done()


def save_vehicle_event(
    track_id,
    vehicle_type,
    vehicle_color,
    timestamp_seconds,
    direction,
    run_id=None,
):
    """
    Queue a crossing event instead of waiting for MySQL.

    run_id is optional for compatibility with any older caller. The updated
    detector should pass the UUID generated at the beginning of each run.
    """
    if _worker is None or not _worker.is_alive():
        raise RuntimeError(
            "Database worker is not running. "
            "Call initialize_database() first."
        )

    event = {
        "run_id": str(run_id) if run_id is not None else None,
        "track_id": int(track_id),
        "vehicle_type": str(vehicle_type),
        "vehicle_color": str(vehicle_color),
        "timestamp_seconds": float(timestamp_seconds),
        "direction": str(direction),
    }

    # Wait for queue space rather than silently discarding a crossing event.
    _event_queue.put(event)


def shutdown_database():
    """Drain pending events, stop the worker, and return failures."""
    global _worker

    if _worker is None:
        return _database_errors

    if _worker.is_alive():
        _event_queue.join()
        _event_queue.put(None)
        _worker.join()

    print("Background database writer stopped.")
    print(f"Database write failures: {_database_errors}")
    return _database_errors

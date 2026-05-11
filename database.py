"""
SQLite database layer - replaces the fragile JSON file approach.
Provides thread-safe access, proper queries, and data integrity.
"""
import sqlite3
import json
import os
from datetime import datetime


class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
        return conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute('''CREATE TABLE IF NOT EXISTS potholes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            road_name TEXT DEFAULT 'Unknown Road',
            severity TEXT DEFAULT 'Minor',
            cost REAL DEFAULT 0,
            source TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        conn.close()

    # ---------- WRITE ----------

    def add_potholes_batch(self, records, source):
        """Insert multiple pothole records at once."""
        conn = self._get_conn()
        count = 0
        for r in records:
            conn.execute(
                'INSERT INTO potholes (lat, lon, road_name, severity, cost, source) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (r.get('lat', 0), r.get('lon', 0),
                 r.get('road_name', 'Unknown Road'),
                 r.get('severity', 'Minor'),
                 r.get('cost', 0), source)
            )
            count += 1
        conn.commit()
        conn.close()
        return count

    # ---------- READ ----------

    def get_all(self):
        conn = self._get_conn()
        rows = conn.execute(
            'SELECT * FROM potholes ORDER BY created_at DESC'
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_by_road(self, road_name):
        conn = self._get_conn()
        rows = conn.execute(
            'SELECT * FROM potholes WHERE road_name = ? ORDER BY created_at DESC',
            (road_name,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_unique_roads(self):
        conn = self._get_conn()
        rows = conn.execute(
            'SELECT DISTINCT road_name FROM potholes ORDER BY road_name'
        ).fetchall()
        conn.close()
        return [r['road_name'] for r in rows]

    def get_nearby(self, lat, lon, radius_m=500):
        """Approximate nearby search (1 degree ≈ 111 km)."""
        deg = radius_m / 111_000
        conn = self._get_conn()
        rows = conn.execute(
            'SELECT * FROM potholes WHERE ABS(lat - ?) < ? AND ABS(lon - ?) < ?',
            (lat, deg, lon, deg)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_stats(self):
        conn = self._get_conn()
        total = conn.execute('SELECT COUNT(*) c FROM potholes').fetchone()['c']
        cost = conn.execute(
            'SELECT COALESCE(SUM(cost),0) c FROM potholes'
        ).fetchone()['c']
        zones = conn.execute(
            'SELECT COUNT(DISTINCT road_name) c FROM potholes'
        ).fetchone()['c']

        sev = {}
        for row in conn.execute(
            'SELECT severity, COUNT(*) c FROM potholes GROUP BY severity'
        ):
            sev[row['severity']] = row['c']

        conn.close()
        return {
            'total_potholes': total,
            'total_cost': cost,
            'critical_zones': zones,
            'severity_breakdown': sev
        }

    # ---------- DELETE ----------

    def reset(self):
        conn = self._get_conn()
        conn.execute('DELETE FROM potholes')
        conn.commit()
        conn.close()

    # ---------- MIGRATION ----------

    def import_from_json(self, json_path):
        """One-time import from the legacy pothole_db.json file."""
        if not os.path.exists(json_path):
            return 0
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            if not data:
                return 0
            return self.add_potholes_batch(data, 'json_import')
        except Exception:
            return 0

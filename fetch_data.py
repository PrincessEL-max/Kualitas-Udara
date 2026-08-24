"""
fetch_data.py — Ambil data historis kualitas udara Jakarta dari OpenAQ API v3
Bagian dari proyek portofolio data science: prediksi & analisis AQI Jakarta.

SEBELUM RUN:
1. Daftar API key gratis di https://explore.openaq.org (klik "Get API Key" /
   sign up), OpenAQ v3 mewajibkan API key di setiap request.
2. Set sebagai environment variable:
     export OPENAQ_API_KEY="apikey_kamu_disini"
3. Install dependencies:
     pip install -r requirements.txt
4. Jalankan:
     python fetch_data.py

CATATAN: Skrip ini belum pernah dites live (sandbox Claude gak bisa akses
api.openaq.org). Struktur endpoint & nama parameter query di bawah ini
diambil dari dokumentasi resmi docs.openaq.org (Agustus 2026). Kalau field
respons ternyata beda (misal struktur "period"), print(payload) mentahnya
dulu terus kabarin aku — gampang disesuaikan.
"""

import os
import time
from datetime import date

import pandas as pd
import requests
API_BASE = "https://api.openaq.org/v3"
API_KEY = os.environ.get("OPENAQ_API_KEY")
if not API_KEY:
    raise SystemExit(
        "OPENAQ_API_KEY belum di-set.\n"
        "Daftar API key gratis di https://explore.openaq.org lalu jalankan:\n"
        "  export OPENAQ_API_KEY='apikey_kamu'"
    )

HEADERS = {"X-API-Key": API_KEY}

# Bounding box DKI Jakarta: minLon,minLat,maxLon,maxLat
# (mencakup Jakarta Pusat/Utara/Barat/Selatan/Timur, gak termasuk Kep. Seribu)
JAKARTA_BBOX = "106.68,-6.37,106.97,-6.07"

# Rentang tanggal data historis yang ditarik — sesuaikan kalau mau range lain
DATE_FROM = "2023-01-01"
DATE_TO = date.today().isoformat()

# Parameter target. Opsi lain yang biasanya ada: "pm10", "no2", "o3", "co", "so2"
TARGET_PARAMETER = "pm25"

PAGE_LIMIT = 1000  # limit maksimum per halaman menurut docs OpenAQ


def get_jakarta_locations():
    """Ambil semua stasiun pemantauan di dalam bounding box Jakarta."""
    locations = []
    page = 1
    while True:
        resp = requests.get(
            f"{API_BASE}/locations",
            headers=HEADERS,
            params={"bbox": JAKARTA_BBOX, "limit": PAGE_LIMIT, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        locations.extend(results)
        if len(results) < PAGE_LIMIT:
            break
        page += 1
        time.sleep(0.5)  # jaga-jaga rate limit
    return locations


def find_target_sensors(locations, parameter_name=TARGET_PARAMETER):
    """Dari daftar lokasi, ambil semua sensor yang mengukur parameter target."""
    sensors = []
    for loc in locations:
        coords = loc.get("coordinates") or {}
        for sensor in loc.get("sensors", []):
            if sensor.get("parameter", {}).get("name") == parameter_name:
                sensors.append(
                    {
                        "sensor_id": sensor["id"],
                        "location_id": loc["id"],
                        "location_name": loc["name"],
                        "latitude": coords.get("latitude"),
                        "longitude": coords.get("longitude"),
                    }
                )
    return sensors


def fetch_daily_measurements(sensor_id, date_from=DATE_FROM, date_to=DATE_TO):
    """Ambil rata-rata harian satu sensor, dengan pagination."""
    all_rows = []
    page = 1
    while True:
        resp = requests.get(
            f"{API_BASE}/sensors/{sensor_id}/measurements/daily",
            headers=HEADERS,
            params={
                "datetime_from": date_from,
                "datetime_to": date_to,
                "limit": PAGE_LIMIT,
                "page": page,
            },
            timeout=30,
        )
        if resp.status_code == 408:
            print(f"    Timeout ambil sensor {sensor_id} halaman {page}, skip.")
            break
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        all_rows.extend(results)
        if len(results) < PAGE_LIMIT:
            break
        page += 1
        time.sleep(0.5)
    return all_rows


def main():
    os.makedirs("data", exist_ok=True)

    print("1. Mengambil daftar stasiun pemantauan di Jakarta...")
    locations = get_jakarta_locations()
    print(f"   Ditemukan {len(locations)} stasiun.")

    pd.DataFrame(
        [
            {
                "location_id": loc["id"],
                "name": loc["name"],
                "provider": (loc.get("provider") or {}).get("name"),
                "is_monitor": loc.get("isMonitor"),
                "latitude": (loc.get("coordinates") or {}).get("latitude"),
                "longitude": (loc.get("coordinates") or {}).get("longitude"),
                "parameters": ", ".join(
                    sorted(
                        {
                            s.get("parameter", {}).get("name", "?")
                            for s in loc.get("sensors", [])
                        }
                    )
                ),
            }
            for loc in locations
        ]
    ).to_csv("data/locations.csv", index=False)
    print("   Disimpan ke data/locations.csv")

    print(f"2. Mencari sensor '{TARGET_PARAMETER}' di antara stasiun tersebut...")
    sensors = find_target_sensors(locations)
    print(f"   Ditemukan {len(sensors)} sensor.")

    print(f"3. Menarik data harian {TARGET_PARAMETER} per sensor ({DATE_FROM} s/d {DATE_TO})...")
    all_measurements = []
    for s in sensors:
        print(f"   - {s['location_name']} (sensor {s['sensor_id']})")
        rows = fetch_daily_measurements(s["sensor_id"])
        for r in rows:
            period = r.get("period", {}) or {}
            datetime_from = (period.get("datetimeFrom") or {}).get("local")
            all_measurements.append(
                {
                    "location_id": s["location_id"],
                    "location_name": s["location_name"],
                    "latitude": s["latitude"],
                    "longitude": s["longitude"],
                    "date": datetime_from,
                    "value": r.get("value"),
                    "parameter": TARGET_PARAMETER,
                }
            )

    df = pd.DataFrame(all_measurements)
    out_path = f"data/{TARGET_PARAMETER}_jakarta_daily.csv"
    df.to_csv(out_path, index=False)
    print(f"Selesai. {len(df)} baris data tersimpan di {out_path}")


if __name__ == "__main__":
    main()

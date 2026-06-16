#!/usr/bin/env python3
"""
Hospital Stock Dashboard - Flask Backend
=========================================
Rules:
  - Pharmacy = supply source, max 100, NEVER runs out (staff keeps it full)
  - ICU and General Ward = destinations, max 100 each
  - Robot dispatches when stock <= 55 AND ML model predicts refill
  - Robot delivers 50 units per trip
  - Pharmacy stock does NOT decrease in simulation
    (it decreases only when robot does a pickup, then resets to 100)
"""

from flask import Flask, render_template, jsonify, request
import joblib, json, os, random, time, threading
from datetime import datetime

app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, '..', 'ml_model')

model    = None
metadata = {}

def load_model():
    global model, metadata
    mp = os.path.join(MODEL_DIR, 'stock_model.pkl')
    jp = os.path.join(MODEL_DIR, 'model_metadata.json')
    if os.path.exists(mp):
        model = joblib.load(mp)
        print("[OK] Model loaded")
    else:
        print("[!!] Model not found — run: python3 ml_model/train_model.py")
    if os.path.exists(jp):
        with open(jp) as f:
            metadata = json.load(f)

load_model()

# ── Room positions matching Gazebo world ───────────────────────
ROOMS = {
    "pharmacy":     {"x": -13.0, "y":  9.0, "room_type": 2},
    "icu":          {"x": -12.0, "y":  0.5, "room_type": 1},
    "general_ward": {"x":  10.0, "y": -5.0, "room_type": 0},
}

DISPATCH_THR  = 55   # dashboard warns AND robot dispatches at this level
CRITICAL_THR  = 30   # true emergency
PHARMACY_MAX  = 100   # pharmacy always kept at this level by staff

# ── Shared state ───────────────────────────────────────────────
state = {
    "robot": {
        "x":           -13.0,
        "y":             9.0,
        "status":       "idle",
        "current_task":  None,
        "path":          [],
        "carrying":      0,       # units robot is currently holding
    },
    "stocks": {
        "pharmacy": {
            "name":          "Pharmacy",
            "icon":          "💊",
            "role":          "source",
            "current_stock": 100,
            "max_stock":     100,
            "refill_needed": False,
            "last_delivery": None,
            "room_type":      2,
            "alert_level":   "ok",
        },
        "icu": {
            "name":          "ICU",
            "icon":          "🏥",
            "role":          "destination",
            "current_stock":  52,
            "max_stock":     100,
            "refill_needed": False,
            "last_refill":   None,
            "room_type":      1,
            "alert_level":   "ok",
        },
        "general_ward": {
            "name":          "General Ward",
            "icon":          "🛏️",
            "role":          "destination",
            "current_stock":  70,
            "max_stock":     100,
            "refill_needed": False,
            "last_refill":   None,
            "room_type":      0,
            "alert_level":   "ok",
        },
    },
    "alerts":          [],
    "prediction_log":  [],
    "last_updated":    datetime.now().isoformat(),
}


def predict_refill(room_key, current_stock):
    """Run ML model prediction for a destination room."""
    if model is None:
        needed = current_stock <= DISPATCH_THR
        return needed, 0.9 if needed else 0.1

    now       = datetime.now()
    room_type = ROOMS.get(room_key, {}).get("room_type", 0)
    shift     = 0 if 6 <= now.hour < 14 else (1 if 14 <= now.hour < 22 else 2)

    features = [[
        now.hour,
        random.randint(8, 25),
        current_stock,
        shift,
        random.randint(0, 3) if room_type == 1 else random.randint(0, 1),
        round(max(1.0, (100 - current_stock) / 10.0), 1),
        now.weekday() + 1,
        room_type,
    ]]

    try:
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        return bool(pred), round(float(prob), 3)
    except Exception:
        return current_stock <= DISPATCH_THR, 0.9


def simulate():
    """
    Background thread:
    - Drains ICU and General Ward stock slowly (simulates usage)
    - Pharmacy stays full (staff maintain it)
    - Runs ML prediction and updates alert levels
    """
    while True:
        time.sleep(5)

        for room_key, s in state["stocks"].items():

            # PHARMACY — never drain, always keep at max
            if s["role"] == "source":
                s["current_stock"] = PHARMACY_MAX
                s["alert_level"]   = "ok"
                continue

            # Skip if robot is currently delivering here
            if (state["robot"]["status"] == "refilling" and
                    state["robot"]["current_task"] == room_key):
                continue

            # Drain rate — kept slow so stock doesn't plummet past
            # the dispatch threshold (55) between robot's 5s checks.
            # ICU: ~0.5 units/5s, General Ward: ~0.3 units/5s
            rate  = {"icu": 0.5, "general_ward": 0.3}
            drain = random.uniform(0.1, rate.get(room_key, 0.3))
            s["current_stock"] = max(0.0, round(s["current_stock"] - drain, 1))

            # ML prediction
            refill_needed, conf = predict_refill(room_key, s["current_stock"])
            s["refill_needed"] = refill_needed

            # Alert level
            if s["current_stock"] < CRITICAL_THR:
                s["alert_level"] = "critical"
            elif s["current_stock"] <= DISPATCH_THR:
                s["alert_level"] = "warning"
            else:
                s["alert_level"] = "ok"

            # Log early predictions (stock still above 30)
            if refill_needed and conf > 0.55:
                early = s["current_stock"] > CRITICAL_THR
                entry = {
                    "time":       datetime.now().strftime("%H:%M:%S"),
                    "room":       s["name"],
                    "stock":      round(s["current_stock"], 1),
                    "confidence": f"{conf*100:.0f}%",
                    "action":     "Dispatching robot (predicted early)"
                                  if early else "!! Already critical",
                    "early":      early,
                }
                state["prediction_log"].insert(0, entry)
                state["prediction_log"] = state["prediction_log"][:20]

        state["last_updated"] = datetime.now().isoformat()


threading.Thread(target=simulate, daemon=True).start()


# ── Routes ─────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/state')
def get_state():
    return jsonify(state)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    room_key      = data.get("room", "icu")
    current_stock = data.get("current_stock", 50)

    if state["stocks"].get(room_key, {}).get("role") == "source":
        return jsonify({
            "refill_needed": False,
            "confidence":    0.0,
            "note":          "Pharmacy is source — never needs refill"
        })

    refill_needed, confidence = predict_refill(room_key, current_stock)

    if room_key in state["stocks"]:
        state["stocks"][room_key]["current_stock"] = current_stock
        state["stocks"][room_key]["refill_needed"]  = refill_needed

    return jsonify({
        "room":              room_key,
        "refill_needed":     refill_needed,
        "confidence":        confidence,
        "current_stock":     current_stock,
        "dispatch_threshold": DISPATCH_THR,
        "critical_threshold": CRITICAL_THR,
    })


@app.route('/api/robot_update', methods=['POST'])
def robot_update():
    """
    Called by robot node every 0.5s with position + status.
    Status values:
      moving   — robot travelling
      pickup   — robot loading at pharmacy (pharmacy resets to 100 after)
      refilled — robot delivered to ICU or Ward (destination stock increases)
      idle     — robot at rest at pharmacy
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    state["robot"]["x"]            = data.get("x",       -13.0)
    state["robot"]["y"]            = data.get("y",         9.0)
    state["robot"]["status"]       = data.get("status",  "idle")
    state["robot"]["current_task"] = data.get("task",      None)
    state["robot"]["path"]         = data.get("path",       [])

    # Robot picked up from pharmacy → now carrying supplies
    if data.get("status") == "pickup":
        load = data.get("load", 50)
        state["robot"]["carrying"] = load          # robot loaded up
        ph = state["stocks"]["pharmacy"]
        ph["current_stock"]  = PHARMACY_MAX
        ph["last_delivery"]  = datetime.now().strftime("%H:%M:%S")

    # Robot delivered to destination → carrying drops to 0
    elif data.get("status") == "refilled":
        room = data.get("task")
        load = data.get("load", 50)
        state["robot"]["carrying"] = 0             # robot unloaded
        if room and room in state["stocks"]:
            s = state["stocks"][room]
            if s["role"] == "destination":
                s["current_stock"] = min(s["max_stock"],
                                         s["current_stock"] + load)
                s["refill_needed"] = False
                s["alert_level"]   = "ok"
                s["last_refill"]   = datetime.now().strftime("%H:%M:%S")
                state["robot"]["status"] = "idle"

                state["alerts"].insert(0, {
                    "time":    datetime.now().strftime("%H:%M:%S"),
                    "message": f"Delivered {load} units to {s['name']}",
                    "type":    "success",
                })
                state["alerts"] = state["alerts"][:10]

    # Robot idle → carrying = 0
    elif data.get("status") == "idle":
        state["robot"]["carrying"] = 0

    return jsonify({"ok": True})


@app.route('/api/model_info')
def model_info():
    return jsonify(metadata)


if __name__ == '__main__':
    print("=" * 50)
    print(" Hospital Stock Dashboard")
    print(" Open: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)

"""
AI24x7 Predictive Analytics - Theft Forecasting
Uses historical CCTV data to predict when/where theft is likely.
Market's FIRST proactive AI security system.
"""
import os, json, sqlite3, numpy as np, threading, time
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# ─── Baseline Learning ──────────────────────
class PatternLearner:
    """
    Learns normal patterns from 30+ days of CCTV data.
    Builds baseline: what is "normal" for each camera/zone/time.
    """
    
    def __init__(self, camera_id, min_days=30):
        self.camera_id = camera_id
        self.min_days = min_days
        self.baseline = {}
        self.hourly_patterns = defaultdict(list)
        self.day_patterns = defaultdict(list)
        self.weekly_patterns = defaultdict(list)
        self.weather_correlation = {}
    
    def ingest_alert_log(self, alert_data):
        """
        Process historical alert data.
        alert_data: list of {timestamp, alert_type, location, ...}
        """
        for alert in alert_data:
            ts = datetime.fromisoformat(alert.get("timestamp", datetime.now().isoformat()))
            
            hour = ts.hour
            day = ts.weekday()  # 0=Monday
            is_weekend = day >= 5
            is_night = hour >= 22 or hour <= 5
            
            key = (hour, "weekday" if not is_weekend else "weekend")
            self.hourly_patterns[key].append(alert)
    
    def calculate_baseline(self):
        """Calculate normal activity baseline"""
        for key, alerts in self.hourly_patterns.items():
            hour, period = key
            count = len(alerts)
            
            self.baseline[f"{period}_{hour}"] = {
                "avg_alerts": count / max(1, self.min_days),
                "max_alerts": max((a.get("confidence", 0.5) for a in alerts), default=0),
                "sample_count": len(alerts)
            }
    
    def predict_risk(self, hour=None, day=None, weather=None):
        """
        Predict risk score for given conditions.
        Returns: float 0-100 (risk percentage)
        """
        hour = hour or datetime.now().hour
        day = day or datetime.now().weekday()
        period = "weekend" if day >= 5 else "weekday"
        key = f"{period}_{hour}"
        
        base = self.baseline.get(key, {"avg_alerts": 0})["avg_alerts"]
        
        # Risk modifiers
        risk = 20  # Base risk
        
        # Time-based
        if hour >= 22 or hour <= 3:  # Late night
            risk += 25
        if hour >= 1 and hour <= 4:  # Deep night
            risk += 15
        
        # Day-based
        if day >= 5:  # Weekend
            risk += 10
        
        # Baseline anomaly
        if base > 3:
            risk += 20
        
        # Weather correlation
        if weather:
            if weather.get("rain"):
                risk += 15
            if weather.get("fog"):
                risk += 10
        
        return min(risk, 100)


# ─── Predictive Engine ──────────────────────
class PredictiveEngine:
    """
    Main predictive analytics engine.
    Analyzes patterns, generates forecasts, sends alerts.
    """
    
    def __init__(self, db_path="/opt/ai24x7/ai24x7_super_admin.db"):
        self.db_path = db_path
        self.cameras = {}
        self.learners = {}
        self.weather_cache = None
        self.last_fetch = None
    
    def add_camera(self, camera_id, location=""):
        learner = PatternLearner(camera_id)
        self.learners[camera_id] = learner
        self.cameras[camera_id] = {"location": location, "risk": 0}
    
    def load_historical_data(self, days=30):
        """Load 30+ days of alert history for pattern learning"""
        if not os.path.exists(self.db_path):
            return
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        rows = conn.execute("""
            SELECT created_at, type, severity, camera_name, confidence
            FROM alerts
            WHERE created_at >= ?
        """, (cutoff,)).fetchall()
        
        for row in rows:
            camera = row["camera_name"] or "default"
            if camera in self.learners:
                self.learners[camera].ingest_alert_log([{
                    "timestamp": row["created_at"],
                    "type": row["type"],
                    "severity": row["severity"],
                    "confidence": row.get("confidence", 0.5)
                }])
        
        for learner in self.learners.values():
            learner.calculate_baseline()
        
        conn.close()
        print(f"📊 Loaded {len(rows)} historical alerts for pattern learning")
    
    def fetch_weather(self):
        """Fetch current weather for risk correlation"""
        try:
            import requests
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast"
                "?latitude=25.59&longitude=82.99&current=rain,weather_code",
                timeout=10
            )
            data = r.json().get("current", {})
            self.weather_cache = {
                "rain": data.get("rain", 0) > 0,
                "weather_code": data.get("weather_code", 0)
            }
            self.last_fetch = datetime.now()
            return self.weather_cache
        except:
            return {}
    
    def predict_all(self):
        """Generate risk predictions for all cameras"""
        if not self.weather_cache or (datetime.now() - self.last_fetch).seconds > 1800:
            self.fetch_weather()
        
        predictions = {}
        for camera_id, camera_info in self.cameras.items():
            risk = self.predict_camera(camera_id)
            self.cameras[camera_id]["risk"] = risk
            
            predictions[camera_id] = {
                "location": camera_info["location"],
                "risk_score": risk,
                "risk_level": self._risk_level(risk),
                "factors": self._explain_risk(camera_id)
            }
        
        return predictions
    
    def predict_camera(self, camera_id):
        """Predict risk for specific camera"""
        if camera_id not in self.learners:
            return 20  # Default low risk
        
        learner = self.learners[camera_id]
        weather = self.weather_cache or {}
        now = datetime.now()
        
        risk = learner.predict_risk(now.hour, now.weekday(), weather)
        
        return min(risk, 100)
    
    def _risk_level(self, score):
        if score >= 70:
            return "HIGH"
        elif score >= 45:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _explain_risk(self, camera_id):
        """Explain why risk is high"""
        factors = []
        now = datetime.now()
        
        if now.hour >= 22 or now.hour <= 5:
            factors.append("Night time (+25 risk)")
        if now.weekday() >= 5:
            factors.append("Weekend (+10 risk)")
        if self.weather_cache and self.weather_cache.get("rain"):
            factors.append("Rainy weather (+15 risk)")
        
        learner = self.learners.get(camera_id)
        if learner:
            base = learner.baseline.get(f"{'weekend' if now.weekday()>=5 else 'weekday'}_{now.hour}", {})
            if base.get("avg_alerts", 0) > 3:
                factors.append("High historical activity (+20 risk)")
        
        return factors
    
    def generate_daily_forecast(self):
        """Generate 24-hour forecast for today"""
        now = datetime.now()
        forecast = []
        
        for hour in range(now.hour, now.hour + 24):
            h = hour % 24
            day = now.weekday() if h >= now.hour else (now.weekday() + 1) % 7
            
            total_risk = 0
            count = 0
            for camera_id in self.learners:
                learner = self.learners[camera_id]
                r = learner.predict_risk(h, day, self.weather_cache or {})
                total_risk += r
                count += 1
            
            avg_risk = total_risk / max(count, 1)
            
            risk_str = "🔴 HIGH" if avg_risk >= 70 else "🟡 MEDIUM" if avg_risk >= 45 else "🟢 LOW"
            
            time_str = f"{h:02d}:00"
            am_pm = "AM" if h < 12 else "PM"
            if h == 0:
                time_str = "12 AM"
            elif h < 12:
                time_str = f"{h} AM"
            elif h == 12:
                time_str = "12 PM"
            else:
                time_str = f"{h-12} PM"
            
            forecast.append({
                "time": time_str,
                "hour": h,
                "risk_score": round(avg_risk, 1),
                "risk_level": risk_str,
                "recommendation": self._get_recommendation(avg_risk)
            })
        
        return forecast
    
    def _get_recommendation(self, risk):
        if risk >= 70:
            return "Deploy extra security. Alert police."
        elif risk >= 45:
            return "Increase monitoring sensitivity."
        else:
            return "Normal operations."


# ─── Forecast Alert Generator ──────────────
class ForecastAlertGenerator:
    """Generates proactive alerts based on predictions"""
    
    def __init__(self, engine):
        self.engine = engine
        self.last_alerts = {}
        self.alert_cooldown = 3600  # 1 hour
    
    def check_and_alert(self):
        """Check predictions and send proactive alerts if needed"""
        predictions = self.engine.predict_all()
        
        alerts_to_send = []
        now = time.time()
        
        for camera_id, pred in predictions.items():
            risk = pred["risk_score"]
            last = self.last_alerts.get(camera_id, 0)
            
            if risk >= 70 and now - last > self.alert_cooldown:
                alerts_to_send.append({
                    "type": "high_risk_forecast",
                    "camera_id": camera_id,
                    "location": pred["location"],
                    "risk_score": risk,
                    "timestamp": datetime.now().isoformat(),
                    "factors": pred["factors"],
                    "message": (
                        f"⚠️ HIGH RISK PREDICTED!\n"
                        f"Location: {pred['location']}\n"
                        f"Risk Score: {risk:.0f}%\n"
                        f"Time: {datetime.now().strftime('%I:%M %p')}\n"
                        f"Factors: {', '.join(pred['factors'])}\n"
                        f"Recommendation: {self._get_recommendation(risk)}"
                    )
                })
                self.last_alerts[camera_id] = now
        
        return alerts_to_send


# ─── Flask API ─────────────────────────────
def create_predictive_api():
    from flask import Flask, jsonify, request
    app = Flask(__name__)
    
    engine = PredictiveEngine()
    forecaster = ForecastAlertGenerator(engine)
    
    @app.route("/predictive/health")
    def health(): return jsonify({"status": "ok"})
    
    @app.route("/predictive/camera/add", methods=["POST"])
    def add_camera():
        data = request.get_json()
        engine.add_camera(data["camera_id"], data.get("location", ""))
        return jsonify({"success": True})
    
    @app.route("/predictive/train", methods=["POST"])
    def train():
        days = request.args.get("days", 30, type=int)
        engine.load_historical_data(days)
        return jsonify({"success": True, "message": f"Trained on {days} days of data"})
    
    @app.route("/predictive/forecast")
    def forecast():
        return jsonify(engine.generate_daily_forecast())
    
    @app.route("/predictive/risk")
    def risk():
        return jsonify(engine.predict_all())
    
    @app.route("/predictive/alert")
    def alert():
        alerts = forecaster.check_and_alert()
        return jsonify({"alerts": alerts})
    
    return app


if __name__ == "__main__":
    import uvicorn
    app = create_predictive_api()
    print("🎯 AI24x7 Predictive Analytics running on port 5065")
    uvicorn.run(app, host="0.0.0.0", port=5065)
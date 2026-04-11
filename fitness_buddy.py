"""
╔══════════════════════════════════════════╗
║         FITNESS BUDDY  🏋️               ║
║   Your Personal Home Workout Coach      ║
║   Team: Rohan, Arhaan & Dhruv           ║
╚══════════════════════════════════════════╝

No gym needed. No trainer needed. Just you, your camera, and FitnessBuddy!

Uses:
  - MediaPipe Pose for joint detection
  - Rule-based angle math for form checking
  - pyttsx3 for voice coaching
  - OpenCV for display

Run: python fitness_buddy.py
"""

import cv2
import numpy as np
import subprocess
import platform
import time
import sys
import os
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# Fix macOS threading crash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# Explicit mediapipe imports for 0.10.30+ compatibility
import mediapipe as mp
import mediapipe as mp

mp_pose_module = mp.solutions.pose
mp_drawing_module = mp.solutions.drawing_utils
PoseLandmark = mp.solutions.pose.PoseLandmark

# ──────────────────────────────────────────────
#  VOICE ENGINE  (non-blocking)
# ──────────────────────────────────────────────
class VoiceCoach:
    """
    Voice coach — no threading at all to avoid macOS mutex crash.
    Uses subprocess.Popen (fire-and-forget) on macOS.
    Uses pyttsx3 on Windows/Linux.
    """
    def __init__(self):
        self._is_mac = platform.system() == 'Darwin'
        self._last_spoken = {}
        self._cooldown = 3.0
        self._proc = None  # track current say process

    def speak(self, text, key=None, cooldown=None):
        cd = cooldown if cooldown is not None else self._cooldown
        now = time.time()
        k = key or text
        if now - self._last_spoken.get(k, 0) < cd:
            return
        self._last_spoken[k] = now

        try:
            if self._is_mac:
                # Kill previous utterance if still running
                if self._proc and self._proc.poll() is None:
                    self._proc.terminate()
                # Fire-and-forget: does NOT block the main thread
                self._proc = subprocess.Popen(
                    ['say', '-r', '175', text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # Windows / Linux
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 165)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception:
                    pass
        except Exception:
            pass


# ──────────────────────────────────────────────
#  MATH HELPERS
# ──────────────────────────────────────────────
def calc_angle(a, b, c):
    """Angle at point B formed by A-B-C (in degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def get_landmark(lm_list, idx, w, h):
    """Return (x, y) in pixel coords."""
    lm = lm_list[idx]
    return int(lm.x * w), int(lm.y * h)


# ──────────────────────────────────────────────
#  DRAWING HELPERS
# ──────────────────────────────────────────────
BG_COLOR   = (15, 15, 25)
GREEN      = (0, 230, 100)
RED        = (0, 60, 220)
YELLOW     = (0, 200, 240)
WHITE      = (240, 240, 240)
ACCENT     = (255, 140, 0)
PANEL_BG   = (30, 30, 45)

def draw_panel(frame, x, y, w, h, color=PANEL_BG, alpha=0.7):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x+w, y+h), ACCENT, 1)

def draw_text(frame, text, pos, size=0.7, color=WHITE, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(frame, text, pos, font, size, (0,0,0), thickness+2)
    cv2.putText(frame, text, pos, font, size, color, thickness)

def draw_progress_bar(frame, x, y, w, h, pct, color=GREEN):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,70), -1)
    fill = int(w * pct)
    cv2.rectangle(frame, (x, y), (x+fill, y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), WHITE, 1)

def draw_angle_arc(frame, center, angle, color=GREEN):
    cv2.circle(frame, center, 18, color, -1)
    draw_text(frame, f"{int(angle)}", (center[0]-14, center[1]+5), 0.45, (0,0,0), 1)


# ──────────────────────────────────────────────
#  EXERCISE ANALYZERS  (pure rule-based math)
# ──────────────────────────────────────────────

class BicepCurlAnalyzer:
    """
    Tracks elbow angle to count bicep curls.
    DOWN  = arm extended  (angle > 155°)
    UP    = arm curled    (angle < 45°)
    Form rules:
      - Elbow should not drift forward (shoulder-elbow-wrist alignment)
      - Full extension required at bottom
    """
    def __init__(self, side='right'):
        self.side = side
        self.count = 0
        self.stage = 'down'
        self.form_issues = []

    def analyze(self, lm, w, h, voice):
        mp_pose = PoseLandmark
        if self.side == 'right':
            shoulder = get_landmark(lm, mp_pose.RIGHT_SHOULDER.value, w, h)
            elbow    = get_landmark(lm, mp_pose.RIGHT_ELBOW.value, w, h)
            wrist    = get_landmark(lm, mp_pose.RIGHT_WRIST.value, w, h)
            hip      = get_landmark(lm, mp_pose.RIGHT_HIP.value, w, h)
        else:
            shoulder = get_landmark(lm, mp_pose.LEFT_SHOULDER.value, w, h)
            elbow    = get_landmark(lm, mp_pose.LEFT_ELBOW.value, w, h)
            wrist    = get_landmark(lm, mp_pose.LEFT_WRIST.value, w, h)
            hip      = get_landmark(lm, mp_pose.LEFT_HIP.value, w, h)

        elbow_angle = calc_angle(shoulder, elbow, wrist)
        # Torso lean check
        torso_angle = calc_angle(shoulder, hip, (hip[0], hip[1]-100))

        self.form_issues = []

        # Count logic
        if elbow_angle > 155:
            self.stage = 'down'
        if elbow_angle < 45 and self.stage == 'down':
            self.stage = 'up'
            self.count += 1
            voice.speak(f"{self.count}", key=f"curl_count_{self.count}", cooldown=0)

        # Form feedback
        if self.stage == 'down' and elbow_angle > 175:
            pass  # perfect extension
        elif self.stage == 'down' and elbow_angle < 140:
            self.form_issues.append("Fully extend arm!")
            voice.speak("Fully extend your arm at the bottom", key="curl_extend", cooldown=4)

        # Elbow drift: elbow should stay near torso (x close to shoulder x)
        elbow_drift = abs(elbow[0] - shoulder[0])
        if elbow_drift > w * 0.12:
            self.form_issues.append("Keep elbow tucked!")
            voice.speak("Keep your elbow tucked in", key="curl_elbow", cooldown=4)

        pct = max(0, min(1, (160 - elbow_angle) / 120))
        return elbow_angle, elbow, wrist, pct, self.form_issues


class SquatAnalyzer:
    """
    Tracks knee angle to count squats.
    STANDING = knee angle > 160°
    SQUAT    = knee angle < 90°
    Form rules:
      - Knee should not go beyond toes (knee x vs ankle x)
      - Back should stay relatively upright (hip-shoulder angle)
      - Depth: thighs must be at least parallel (knee angle ≤ 90°)
    """
    def __init__(self):
        self.count = 0
        self.stage = 'up'
        self.form_issues = []

    def analyze(self, lm, w, h, voice):
        mp_pose = PoseLandmark
        # Use right side landmarks
        hip    = get_landmark(lm, mp_pose.RIGHT_HIP.value, w, h)
        knee   = get_landmark(lm, mp_pose.RIGHT_KNEE.value, w, h)
        ankle  = get_landmark(lm, mp_pose.RIGHT_ANKLE.value, w, h)
        shoulder = get_landmark(lm, mp_pose.RIGHT_SHOULDER.value, w, h)

        knee_angle = calc_angle(hip, knee, ankle)
        back_angle = calc_angle(hip, shoulder, (shoulder[0], shoulder[1]-100))

        self.form_issues = []

        # Count logic
        if knee_angle > 160:
            self.stage = 'up'
        if knee_angle < 90 and self.stage == 'up':
            self.stage = 'down'
            self.count += 1
            voice.speak(f"{self.count}", key=f"squat_count_{self.count}", cooldown=0)

        # Form: knee over toes
        knee_over_toes = knee[0] - ankle[0]  # positive = knee forward of ankle
        if knee_over_toes > w * 0.07:
            self.form_issues.append("Knee past toes! Push hips back")
            voice.speak("Push your hips back, knee past toes", key="squat_knee", cooldown=4)

        # Form: back angle (lean)
        if back_angle < 55:
            self.form_issues.append("Chest up! Don't lean too far")
            voice.speak("Keep your chest up", key="squat_back", cooldown=4)

        # Depth encouragement
        if self.stage == 'up' and 90 < knee_angle < 130:
            self.form_issues.append("Go deeper! Aim for 90°")
            voice.speak("Go lower for a full squat", key="squat_depth", cooldown=5)

        pct = max(0, min(1, (160 - knee_angle) / 80))
        return knee_angle, knee, ankle, pct, self.form_issues


class PushupAnalyzer:
    """
    Tracks elbow angle to count push-ups (camera from side).
    Also works front-facing by tracking shoulder-elbow-wrist.
    DOWN = elbow angle < 90°
    UP   = elbow angle > 155°
    Form rules:
      - Body must be straight (hip sag or raise)
      - Full extension at top
      - Chest near ground at bottom
    """
    def __init__(self):
        self.count = 0
        self.stage = 'up'
        self.form_issues = []

    def analyze(self, lm, w, h, voice):
        mp_pose = PoseLandmark
        shoulder = get_landmark(lm, mp_pose.RIGHT_SHOULDER.value, w, h)
        elbow    = get_landmark(lm, mp_pose.RIGHT_ELBOW.value, w, h)
        wrist    = get_landmark(lm, mp_pose.RIGHT_WRIST.value, w, h)
        hip      = get_landmark(lm, mp_pose.RIGHT_HIP.value, w, h)
        ankle    = get_landmark(lm, mp_pose.RIGHT_ANKLE.value, w, h)

        elbow_angle = calc_angle(shoulder, elbow, wrist)
        # Body alignment: shoulder-hip-ankle should be ~180°
        body_angle = calc_angle(shoulder, hip, ankle)

        self.form_issues = []

        # Count logic
        if elbow_angle > 155:
            self.stage = 'up'
        if elbow_angle < 90 and self.stage == 'up':
            self.stage = 'down'
            self.count += 1
            voice.speak(f"{self.count}", key=f"push_count_{self.count}", cooldown=0)

        # Form: body straight
        if body_angle < 155:
            self.form_issues.append("Keep body straight! Don't sag hips")
            voice.speak("Keep your body in a straight line", key="push_body", cooldown=4)
        if body_angle > 200:
            self.form_issues.append("Lower your hips!")
            voice.speak("Lower your hips", key="push_hips", cooldown=4)

        # Form: full extension
        if self.stage == 'up' and elbow_angle < 145:
            self.form_issues.append("Fully extend arms at top!")
            voice.speak("Lock your arms at the top", key="push_extend", cooldown=4)

        pct = max(0, min(1, (160 - elbow_angle) / 80))
        return elbow_angle, elbow, wrist, pct, self.form_issues


# ──────────────────────────────────────────────
#  MAIN APP
# ──────────────────────────────────────────────

class FitnessBuddy:
    EXERCISES = ['bicep_curl', 'squat', 'pushup']
    NAMES     = ['💪 BICEP CURL', '🏋 SQUAT', '👊 PUSH-UP']

    def __init__(self):
        self.voice    = VoiceCoach()
        self.mp_pose  = mp_pose_module
        self.mp_draw  = mp_drawing_module
        self.pose     = mp_pose_module.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.ex_idx   = 0
        self.analyzers = {
            'bicep_curl': BicepCurlAnalyzer('right'),
            'squat'     : SquatAnalyzer(),
            'pushup'    : PushupAnalyzer()
        }
        self.tips = {
            'bicep_curl': ["Stand straight", "Curl to chin level", "Slow on the way down", "Keep elbow still"],
            'squat'     : ["Feet shoulder width", "Toes slightly out", "Look forward", "Weight on heels"],
            'pushup'    : ["Hands shoulder wide", "Body like a plank", "Lower chest to ground", "Elbows at 45°"],
        }
        self.tip_timer = time.time()
        self.tip_idx   = 0
        self.session_start = time.time()

    def current_exercise(self):
        return self.EXERCISES[self.ex_idx]

    def switch_exercise(self, direction=1):
        self.ex_idx = (self.ex_idx + direction) % len(self.EXERCISES)
        name = self.NAMES[self.ex_idx]
        self.voice.speak(f"Switching to {name.split()[-1]}", cooldown=0)

    def reset_current(self):
        ex = self.current_exercise()
        if ex == 'bicep_curl':
            self.analyzers['bicep_curl'] = BicepCurlAnalyzer('right')
        elif ex == 'squat':
            self.analyzers['squat'] = SquatAnalyzer()
        elif ex == 'pushup':
            self.analyzers['pushup'] = PushupAnalyzer()
        self.voice.speak("Counter reset", cooldown=0)

    def draw_hud(self, frame, count, stage, pct, form_issues, angle, ex_name):
        h, w = frame.shape[:2]

        # ── Left panel: exercise name + count ──
        draw_panel(frame, 10, 10, 230, 130)
        draw_text(frame, ex_name, (20, 40), 0.6, ACCENT, 2)
        draw_text(frame, str(count), (20, 110), 3.0, GREEN if not form_issues else RED, 4)

        # ── Right panel: stage + angle ──
        draw_panel(frame, w-170, 10, 160, 90)
        draw_text(frame, f"STAGE: {stage.upper()}", (w-165, 40), 0.55, WHITE)
        draw_text(frame, f"ANGLE: {int(angle)}", (w-165, 75), 0.55, YELLOW)

        # ── Progress bar ──
        bar_y = h - 60
        draw_text(frame, "RANGE", (10, bar_y - 8), 0.5, WHITE)
        draw_progress_bar(frame, 10, bar_y, w-20, 20, pct, GREEN if pct < 0.85 else ACCENT)

        # ── Form feedback ──
        if form_issues:
            draw_panel(frame, 10, 150, 320, 30 + 28*len(form_issues), (20, 20, 60), 0.8)
            draw_text(frame, "⚠ FORM ALERT", (20, 172), 0.55, RED, 2)
            for i, issue in enumerate(form_issues[:3]):
                draw_text(frame, f"  • {issue}", (20, 200 + i*28), 0.5, YELLOW)
        else:
            draw_panel(frame, 10, 150, 220, 35, (20, 50, 20), 0.8)
            draw_text(frame, "✓ GREAT FORM!", (20, 175), 0.55, GREEN, 2)

        # ── Rotating tip ──
        tips = self.tips[self.current_exercise()]
        if time.time() - self.tip_timer > 5:
            self.tip_idx = (self.tip_idx + 1) % len(tips)
            self.tip_timer = time.time()
        draw_panel(frame, 10, h-100, 400, 35, PANEL_BG, 0.7)
        draw_text(frame, f"TIP: {tips[self.tip_idx]}", (20, h-77), 0.5, WHITE)

        # ── Session time ──
        elapsed = int(time.time() - self.session_start)
        mm, ss = divmod(elapsed, 60)
        draw_panel(frame, w-140, h-50, 135, 38, PANEL_BG, 0.7)
        draw_text(frame, f"⏱ {mm:02d}:{ss:02d}", (w-130, h-25), 0.6, ACCENT)

        # ── Controls legend ──
        legend = "[A/D] Switch  [R] Reset  [Q] Quit"
        draw_text(frame, legend, (w//2-200, h-15), 0.45, (160,160,160), 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera. Check connection.")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        self.voice.speak("Welcome to Fitness Buddy! Let's work out!", cooldown=0)
        time.sleep(1.5)
        self.voice.speak("Starting with bicep curls. Press A or D to switch exercises.", cooldown=0)

        print("\n" + "="*50)
        print("  FITNESS BUDDY  🏋️  — Controls")
        print("="*50)
        print("  A  : Previous exercise")
        print("  D  : Next exercise")
        print("  R  : Reset counter")
        print("  Q  : Quit")
        print("="*50 + "\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Dark background overlay for better visibility
            overlay = np.zeros_like(frame)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            angle  = 0
            pct    = 0.0
            issues = []
            stage  = 'ready'

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Draw skeleton
                self.mp_draw.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose_module.POSE_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=ACCENT, thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=WHITE, thickness=2)
                )

                ex = self.current_exercise()
                if ex == 'bicep_curl':
                    angle, elbow, wrist, pct, issues = self.analyzers['bicep_curl'].analyze(lm, w, h, self.voice)
                    stage = self.analyzers['bicep_curl'].stage
                    count = self.analyzers['bicep_curl'].count
                    draw_angle_arc(frame, elbow, angle, GREEN if not issues else RED)
                elif ex == 'squat':
                    angle, knee, ankle, pct, issues = self.analyzers['squat'].analyze(lm, w, h, self.voice)
                    stage = self.analyzers['squat'].stage
                    count = self.analyzers['squat'].count
                    draw_angle_arc(frame, knee, angle, GREEN if not issues else RED)
                elif ex == 'pushup':
                    angle, elbow, wrist, pct, issues = self.analyzers['pushup'].analyze(lm, w, h, self.voice)
                    stage = self.analyzers['pushup'].stage
                    count = self.analyzers['pushup'].count
                    draw_angle_arc(frame, elbow, angle, GREEN if not issues else RED)
            else:
                ex    = self.current_exercise()
                count = self.analyzers[ex].count
                draw_panel(frame, w//2-150, h//2-30, 300, 55, (40,0,0), 0.85)
                draw_text(frame, "No person detected!", (w//2-140, h//2+8), 0.7, RED, 2)

            self.draw_hud(frame, count, stage, pct, issues, angle, self.NAMES[self.ex_idx])
            cv2.imshow("FITNESS BUDDY", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('d'):
                self.switch_exercise(1)
            elif key == ord('a'):
                self.switch_exercise(-1)
            elif key == ord('r'):
                self.reset_current()

        # Session summary
        elapsed = int(time.time() - self.session_start)
        mm, ss  = divmod(elapsed, 60)
        bc = self.analyzers['bicep_curl'].count
        sq = self.analyzers['squat'].count
        pu = self.analyzers['pushup'].count
        print(f"\n{'='*50}")
        print(f"  SESSION COMPLETE  ⏱ {mm:02d}:{ss:02d}")
        print(f"  💪 Bicep Curls : {bc}")
        print(f"  🏋 Squats      : {sq}")
        print(f"  👊 Push-ups    : {pu}")
        print(f"{'='*50}\n")
        self.voice.speak(
            f"Great session! You did {bc} curls, {sq} squats, and {pu} push-ups. Amazing work!",
            cooldown=0
        )
        time.sleep(3)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = FitnessBuddy()
    app.run()
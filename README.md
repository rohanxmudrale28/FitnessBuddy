# 🏋️ FITNESS BUDDY
### Your Personal Home Workout Coach
**Team: Rohan, Arhaan & Dhruv**

---

## What is Fitness Buddy?
A computer vision-based fitness coach that tracks your workouts **in real-time** using just your webcam. No gym membership. No trainer. No excuses.

### Features
| Feature | How it works |
|---|---|
| 💪 Bicep Curl Counter | Measures elbow angle, counts full reps |
| 🏋 Squat Counter | Tracks knee angle, ensures full depth |
| 👊 Push-up Counter | Monitors elbow & body alignment |
| 🎙 Voice Coaching | Speaks rep counts and form corrections |
| ⚠ Form Alerts | Real-time alerts for bad form |
| ⏱ Session Timer | Tracks your total workout time |

---

## Tech Stack (No AI/ML — Pure Math!)
- **MediaPipe Pose** — detects 33 body landmarks using your webcam
- **OpenCV** — processes video frames
- **NumPy** — calculates joint angles using trigonometry (cosine rule)
- **pyttsx3** — offline text-to-speech for voice coaching
- **Rule-based logic** — if-else angle thresholds (no machine learning)

### How the angle math works:
```
Given 3 joint positions A, B, C (pixels):
  vector BA = A - B
  vector BC = C - B
  angle = arccos( dot(BA, BC) / (|BA| × |BC|) )
```
That's it. Pure high school trigonometry!

---

## File Structure
```
FitnessBuddy/
│
├── fitness_buddy.py     ← Main app (run this!)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── assets/              ← (optional) icons/images
```

---

## Setup & Run

### Step 1: Install Python 3.9+
Download from https://python.org

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run!
```bash
python fitness_buddy.py
```

### Controls
| Key | Action |
|---|---|
| `D` | Next exercise |
| `A` | Previous exercise |
| `R` | Reset rep counter |
| `Q` / `Esc` | Quit |

---

## Form Rules (Rule-Based, No AI)

### Bicep Curl
- ✅ Elbow angle > 155° = arm fully extended (bottom)
- ✅ Elbow angle < 45° = full curl (top)
- ⚠ Elbow drift > 12% frame width → "Keep elbow tucked!"
- ⚠ Incomplete extension → "Fully extend your arm!"

### Squat
- ✅ Knee angle > 160° = standing
- ✅ Knee angle < 90° = full squat depth
- ⚠ Knee past ankle by >7% frame width → "Push hips back!"
- ⚠ Back lean < 55° → "Chest up!"

### Push-up
- ✅ Elbow angle > 155° = top position
- ✅ Elbow angle < 90° = bottom position
- ⚠ Body angle < 155° → "Don't sag your hips!"
- ⚠ Incomplete extension → "Lock your arms at the top!"

---

## Camera Tips
- Use good lighting (face a window or lamp)
- Stand 6-8 feet from the camera
- For squats: camera at hip height, slightly to the side
- For push-ups: camera to the side works best
- For bicep curls: face the camera directly

---

*Built for Interpersonal Skills course presentation. Inspired by Rohan's original curl counter project.*

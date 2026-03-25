# AI Lip Reader — Biometric Authentication via Lip Movement

A biometric authentication system that verifies user identity through **lip movement patterns** and **facial recognition**. Uses cosine similarity to compare lip motion sequences, making it resistant to photo spoofing attacks.

## Demo

```
User registers → records lip movement → face captured
User logs in   → lip movement compared via cosine similarity → face verified → access granted
```

## How It Works

1. **Face detection** — dlib detects 68 facial landmarks in real time
2. **Lip tracking** — extracts 20 lip landmark points per frame
3. **Motion recording** — captures relative lip offsets while moving
4. **Cosine similarity** — compares login sequence against registered pattern
5. **Face verification** — face_recognition library confirms identity

```
Webcam feed → dlib landmarks → lip offsets → cosine similarity → auth decision
                                                    ↑
                                           face_recognition check
```

## Tech Stack

- **Backend** — Flask (Python)
- **Computer Vision** — OpenCV, dlib (68-point facial landmarks)
- **ML / Similarity** — NumPy cosine similarity on lip motion vectors
- **Face Recognition** — face_recognition (dlib-based encodings)
- **Frontend** — HTML/CSS with Tailwind, vanilla JS

## Setup

### Requirements

- Python 3.10+
- Webcam
- `shape_predictor_68_face_landmarks.dat` — download from [dlib model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place in project root

### Install

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`

## Project Structure

```
ai-lip-reader/
├── app.py                  # Flask app, lip tracking, cosine similarity
├── templates/
│   ├── base.html           # Base layout
│   ├── landing.html        # Landing page
│   ├── registration.html   # Register form
│   ├── record_registration.html
│   ├── login.html          # Login form
│   ├── record_login.html   # Lip movement capture
│   └── account.html        # Authenticated dashboard
├── requirements.txt
└── .gitignore
```

## Key Algorithm

```python
def cosine_similarity(seq1, seq2):
    seq1_flat = (seq1 - seq1[0]).flatten()
    seq2_flat = (seq2 - seq2[0]).flatten()
    return np.dot(seq1_flat, seq2_flat) / (
        np.linalg.norm(seq1_flat) * np.linalg.norm(seq2_flat)
    )
```

Lip motion is normalized relative to starting position, making it invariant to face position in frame. Similarity threshold: **0.8**.

## Security Notes

- Passwords hashed with bcrypt
- Face verification prevents photo attacks
- Lip motion adds second biometric factor
- `users.csv` and captured faces excluded from version control

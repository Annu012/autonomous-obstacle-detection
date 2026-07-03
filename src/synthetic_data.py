import cv2, numpy as np
from pathlib import Path

for split in ['train', 'val', 'test']:
    Path(f"data/{split}/images").mkdir(parents=True, exist_ok=True)
    Path(f"data/{split}/labels").mkdir(parents=True, exist_ok=True)
    
    num = 150 if split == 'train' else 30
    for i in range(num):
        img = np.ones((720, 1280, 3), dtype=np.uint8) * 100
        cv2.rectangle(img, (0, 400), (1280, 720), (80, 80, 80), -1)
        cv2.line(img, (640, 400), (640, 720), (255, 255, 255), 3)
        
        labels = []
        if np.random.random() < 0.7:
            x, y, w, h = np.random.randint(500, 900), np.random.randint(200, 500), 150, 120
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 200), -1)
            labels.append(f"0 {(x+w/2)/1280:.4f} {(y+h/2)/720:.4f} {w/1280:.4f} {h/720:.4f}")
        
        cv2.imwrite(f"data/{split}/images/scene_{i:04d}.jpg", img)
        with open(f"data/{split}/labels/scene_{i:04d}.txt", 'w') as f:
            if labels:
                f.write('\n'.join(labels))

print("✅ Data ready")
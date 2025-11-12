# Sketch2Poto

**Sketch2Photo** is a computer vision project inspired by the SIGGRAPH paper *“Sketch2Photo: Internet Image Montage”*.  
It converts user-provided sketches and keywords into a realistic composite image by retrieving and blending real photos from the web.

---

## Features
- **Interactive sketching or upload mode** for scene elements  
- **Automated image scraping** from Google Images using Selenium  
- **Content filtering & background selection** using MeanShift clustering, Mahalanobis distance, and horizon detection  
- **Foreground extraction** using Saliency detection + GrabCut segmentation  
- **Shape matching** between user sketch and candidate photos  
- **Realistic blending** via Poisson seamless cloning and LAB color harmonization  
- **Preview grid output** showing sketches, candidates, and final results  


## How It Works
1. User provides sketches (draw or upload) and descriptive tags.  
2. The program scrapes candidate images for each tag and background.  
3. Backgrounds are filtered based on **color consistency**, **horizon clarity**, and **segmentation complexity**.  
4. Foreground objects are extracted and matched with sketches using **contours** and **shape similarity**.  
5. Selected items are composited on the best background using **Poisson blending** for realism.  


## ⚙️ How to Run
1. Activate environment - venv/Scripts/activate
2. Run the program  - python sketch2image.py

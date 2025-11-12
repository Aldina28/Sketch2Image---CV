from __future__ import annotations
import os
import io
import time
import glob
import hashlib
import requests
import numpy as np
import cv2
import PIL
from PIL import Image, ImageDraw
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import sklearn
from sklearn.cluster import MeanShift, estimate_bandwidth
from matplotlib import pyplot as plt
from skimage import segmentation as seg
from tqdm import tqdm

try:
    from utils import detect_horizon_line
except Exception:
    def detect_horizon_line(gray_img):
        h, w = gray_img.shape[:2]
        sob = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
        mean_rows = np.mean(np.abs(sob), axis=1)
        y = int(np.argmax(mean_rows))
        if mean_rows[y] < 5:  
            raise ValueError("Horizon not found")
        return (0, w, y, y)


def draw_items_interactive(num_elements: int, sizes: list[tuple[int,int]], output_dir: str = "sketch2photo"):
    try:
        from tkinter import Tk, Canvas, Button, YES, BOTH
    except Exception as e:
        raise RuntimeError("Tkinter not available in this environment. Run locally with GUI.") from e

    os.makedirs(output_dir, exist_ok=True)

    for element in range(num_elements):
        width, height = sizes[element]
        white = (255, 255, 255)

        output_image = Image.new("RGB", (width + 100, height + 100), white)
        draw = ImageDraw.Draw(output_image)

        def save_and_close():
            filename = os.path.join(output_dir, f"item{element}.jpg")
            output_image.save(filename, quality=90)
            root.destroy()

        def paint(event):
            x1, y1 = (event.x - 3), (event.y - 3)
            x2, y2 = (event.x + 3), (event.y + 3)
            canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
            draw.line([x1, y1, x2, y2], fill="black", width=5)

        root = Tk()
        root.title(f"Draw item {element}")
        canvas = Canvas(root, width=width + 100, height=height + 100, bg='white')
        canvas.pack(expand=YES, fill=BOTH)
        canvas.bind("<B1-Motion>", paint)

        btn = Button(root, text="Save", command=save_and_close)
        btn.pack()

        root.mainloop()


def fetch_image_urls(query: str, max_links_to_fetch: int, wd: webdriver.Chrome, sleep_between_interactions: int = 1):
    """Fetches image URLs from Google Images"""
    def scroll_to_end(driver):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    search_url = "https://www.google.com/search?tbm=isch&q={q}"
    wd.get(search_url.format(q=query))
    image_urls = set()
    image_count = 0
    results_start = 0

    while image_count < max_links_to_fetch:
        scroll_to_end(wd)
        thumbnail_results = wd.find_elements(
            By.CSS_SELECTOR,
            "img.rg_i, img.YQ4gaf, img.sFlh5c, img.Q4LuWd, img.iPVvYb, img.yWs4tf"
        )
        for img in thumbnail_results:
            src = img.get_attribute("src") or img.get_attribute("data-src")
            if src and src.startswith("http"):
                image_urls.add(src)
        number_results = len(thumbnail_results)
        if number_results == 0:
            time.sleep(2)
            thumbnail_results = wd.find_elements(By.CSS_SELECTOR, "img.rg_i, img.YQ4gaf, img.sFlh5c")
            number_results = len(thumbnail_results)

        print(f"Found: {number_results} thumbnails. Extracting links {results_start}:{number_results}")

        for img in thumbnail_results[results_start:number_results]:
            try:
                img.click()
                time.sleep(2)
            except Exception:
                continue
            time.sleep(1)
            actual_images = wd.find_elements(By.CSS_SELECTOR, 'img.n3VNCb, img.r48jcc, img.sFlh5c, img.YQ4gaf')
            for actual_image in actual_images:
                src = actual_image.get_attribute("src")
                if src and src.startswith("http") and not is_blocked_url(src):
                    image_urls.add(src)

            image_count = len(image_urls)
            if image_count >= max_links_to_fetch:
                print(f"Found {image_count} image links — done.")
                return image_urls

        results_start = number_results
        try:
            load_more = wd.find_element(By.CSS_SELECTOR, ".mye4qd")
            if load_more:
                wd.execute_script("arguments[0].click();", load_more)
                time.sleep(1)
        except Exception:
            print("No more results button / can't load more.")
            break

    print(f"Total collected: {len(image_urls)}")
    return image_urls


def persist_image(folder_path: str, file_name_prefix: str, url: str):
    """Download an image URL and save it as JPEG"""
    folder = os.path.join(folder_path, file_name_prefix)
    os.makedirs(folder, exist_ok=True)
    if is_blocked_url(url):
        return None
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")
        return None

    try:
        image_file = io.BytesIO(resp.content)
        image = Image.open(image_file).convert("RGB")
        if image.width < 200 or image.height < 200 or image.width / image.height > 3 or image.height / image.width > 3:
            return None
        hashed = hashlib.sha1(resp.content).hexdigest()[:10]
        file_path = os.path.join(folder, f"{hashed}.jpg")
        with open(file_path, "wb") as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} as {file_path}")
        return file_path
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")
        return None

def ensure_resample_mode():
    try:
        return Image.Resampling.LANCZOS
    except Exception:
        return Image.ANTIALIAS

BLOCKED_DOMAINS = (
    "istockphoto.com",
    "shutterstock.com",
    "alamy.com",
    "gettyimages",
    "dreamstime.com",
    "depositphotos.com",
    "vectorstock.com",
)


def is_blocked_url(url: str) -> bool:
    try:
        lowered = url.lower()
        return any(domain in lowered for domain in BLOCKED_DOMAINS)
    except Exception:
        return False


def normalize_filled(img_np: np.ndarray) -> np.ndarray:
    if img_np is None:
        return np.zeros((1, 1), dtype=np.uint8)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_np.copy()
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return thresh
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [c], 255)
    return mask

def mahalanobis(x=None, data=None, cov=None):
    x = np.asarray(x, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    x_mu = x - np.mean(data, axis=0)
    if cov is None:
        cov = np.cov(data.T)
    inv_covmat = np.linalg.pinv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    if mahal.ndim == 0:
        return float(mahal)
    return np.diag(mahal)


def main():
    number_elements = int(input("Enter the number of elements you want to add: ").strip())
    tags = []
    sizes = []

    print("For each element, provide width, height (integers) and label.")
    for element in range(number_elements):
        w = int(input(f"Element {element} - Enter width: ").strip())
        h = int(input(f"Element {element} - Enter height: ").strip())
        sizes.append((w, h))
        tag = input(f"Element {element} - Enter label/tag: ").strip()
        tags.append(tag)

    mode = input("Do you want to (d)raw sketches or (u)pload existing ones? (d/u): ").strip().lower()

    os.makedirs("sketch2photo", exist_ok=True)

    if mode == 'd':
        print("Opening drawing canvases. Draw with mouse and click Save to store each item.")
        draw_items_interactive(number_elements, sizes, output_dir="sketch2photo")

    elif mode == 'u':
        print("\nUpload or place your sketch images in the 'sketch2photo/' folder.")
        print("Make sure filenames follow this pattern: item0.jpg, item1.jpg, item2.jpg, ...\n")

        for i in range(number_elements):
            path = input(f"Enter path for sketch image for item {i} (or press Enter to use existing file): ").strip()
            default_path = os.path.join("sketch2photo", f"item{i}.jpg")

            if path:
                if os.path.exists(path):
                    img = Image.open(path).convert("RGB")
                    img.save(default_path, quality=90)
                    print(f"Copied to {default_path}")
                else:
                    print(f"Provided path not found: {path}")
            else:
                if os.path.exists(default_path):
                    print(f"Using existing sketch: {default_path}")
                else:
                    print(f"No existing sketch found for item {i}. Please add it manually.")
    else:
        print("Skipping sketches — make sure 'sketch2photo/item0..itemN.jpg' exist before continuing.")

    background_tag = input("Enter the tag for the background: ").strip()
    tags.append(background_tag)

    image_background = Image.new('RGB', (512, 512), (255, 255, 255))
    desired_locations = []
    for i in range(number_elements):
        img_address = os.path.join("sketch2photo", f"item{i}.jpg")
        if not os.path.exists(img_address):
            x = int(input(f"Enter x coordinate of desired location of item {i}: ").strip())
            y = int(input(f"Enter y coordinate of desired location of item {i}: ").strip())
            desired_locations.append((x, y))
            continue
        curr_image = Image.open(img_address)
        resample = ensure_resample_mode()
        curr_image = curr_image.resize((sizes[i][0] + 100, sizes[i][1] + 100), resample)
        x = int(input(f"Enter x coordinate of desired location of item {i}: ").strip())
        y = int(input(f"Enter y coordinate of desired location of item {i}: ").strip())
        desired_locations.append((x, y))
        image_background.paste(curr_image, (x, y))

    print("Starting Chrome webdriver for image scraping...")
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")


    service = Service(ChromeDriverManager().install())
    wd = webdriver.Chrome(service=service, options=chrome_options)

    images_path = "scraped_photos"
    os.makedirs(images_path, exist_ok=True)

    for query in tags:
        print(f"Searching images for: {query}")
        wd.get("https://www.google.com")
        try:
            search_box = wd.find_element(By.CSS_SELECTOR, "input.gLFyf")
            search_box.clear()
            search_box.send_keys(query)
            search_box.submit()
        except Exception:
            wd.get(f"https://www.google.com/search?tbm=isch&q={query}")

        max_links = 10 if query != background_tag else 12
        links = fetch_image_urls(query, max_links_to_fetch=max_links, wd=wd)
        for url in links:
            persist_image(images_path, query, url)

    wd.quit()
    print("Finished scraping.")

    background_image_list = []
    bg_folder = os.path.join(images_path, background_tag)
    if not os.path.exists(bg_folder):
        print(f"No scraped images for background tag '{background_tag}' found in {bg_folder}.")
    else:
        for filename in glob.glob(os.path.join(bg_folder, "*.jpg")):
            try:
                im = Image.open(filename).convert("RGB")
                background_image_list.append(im)
            except Exception:
                continue

    if not background_image_list:
        print("No background images available — exiting.")
        return
    print("Performing content consistency filtering on backgrounds...")
    mahalanobis_distances = []
    for imgs in tqdm(background_image_list):
        img = np.array(imgs)
        reshape_img = img.reshape(-1, 3).astype(np.float32)
        n_samples = min(len(reshape_img), 2000)
        subsample_idx = np.random.choice(len(reshape_img), n_samples, replace=False)
        subsample = reshape_img[subsample_idx]

        try:
            bandwidth = estimate_bandwidth(subsample, quantile=0.2, n_samples=300)
            if bandwidth <= 0:
                bandwidth = 20
            msc = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            msc.fit(subsample)
            counts = np.bincount(msc.labels_)
            max_count = np.argmax(counts)
            data = subsample[msc.labels_ == max_count]
            indices = np.random.choice(len(reshape_img), size=int(0.01*len(reshape_img)) or 1, replace=False)
            curr_distance = np.array([mahalanobis(x=reshape_img[i], data=data) for i in indices])
            mahalanobis_distances.append(np.mean(curr_distance))
        except Exception:
            mahalanobis_distances.append(0.0)

    mahalanobis_distances = sklearn.preprocessing.minmax_scale(np.array(mahalanobis_distances), feature_range=(0, 1))
    lowest_maha_dist = np.argsort(mahalanobis_distances)[:min(10, len(background_image_list))]
    background1 = [background_image_list[i] for i in lowest_maha_dist]

    print("Filtering backgrounds by simple horizon heuristic and segmentation complexity...")
    selected_images1 = []
    for i, img in enumerate(background1):
        width, height = img.size
        image_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        image_grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        try:
            horizon_x1, horizon_x2, horizon_y1, horizon_y2 = detect_horizon_line(image_grayscale)
            p1 = np.array((horizon_x1, horizon_y1))
            p2 = np.array((horizon_x2, horizon_y2))
            p3 = np.array((width / 2, height))
            d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
            if d <= 0.5 * height:
                selected_images1.append(img)
        except Exception:
            selected_images1.append(img)

    if not selected_images1:
        selected_images1 = background1.copy()

    segment_counts = []
    for img in selected_images1:
        try:
            image_np = np.array(img)
            image_slic = seg.slic(image_np, n_segments=60, start_label=1)
            labels = np.unique(image_slic).size
            segment_counts.append(labels)
        except Exception:
            segment_counts.append(100)

    segment_count_normalized = sklearn.preprocessing.minmax_scale(np.array(segment_counts), feature_range=(0, 1))
    selected_mahalanobis = np.zeros(len(selected_images1))
    combined_cost = segment_count_normalized + (0.3 * selected_mahalanobis)
    least_segment_indices = np.argsort(combined_cost)[:min(5, len(selected_images1))]
    selected_background_images = [selected_images1[i] for i in least_segment_indices]

    print("Selecting best candidate images for each scene item by contour matching...")
    selected_scene_items = []

    for idx in range(number_elements):
        tag = tags[idx]
        folder = os.path.join(images_path, tag)
        image_files = glob.glob(os.path.join(folder, "*.jpg")) if os.path.exists(folder) else []
        scores = []
        extracted_images = []

        for filename in image_files:
            try:
                im = Image.open(filename).convert("RGB")
                img_np = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                img_blur = cv2.GaussianBlur(img_np, (5, 5), 0)
                gray_for_sal = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

                try:
                    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
                    success, saliencyMap = saliency.computeSaliency(img_np)
                    sal_map = (saliencyMap * 255).astype("uint8")
                    threshMap = cv2.threshold(sal_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    rect = cv2.boundingRect(threshMap)
                except Exception:
                    rect = (1, 1, img_np.shape[1] - 2, img_np.shape[0] - 2)

                mask = np.zeros(img_np.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                try:
                    cv2.grabCut(img_np, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

                    kernel = np.ones((5,5), np.uint8)
                    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
                    mask2 = cv2.GaussianBlur(mask2.astype(np.float32), (7,7), 0)
                    mask2 = np.clip(mask2, 0, 1)
                    extracted = (img_np * mask2[:, :, None]).astype(np.uint8)
                except Exception:
                    extracted = img_np.copy()

                contour_current = normalize_filled(extracted)

                sketch_path = os.path.join("sketch2photo", f"item{idx}.jpg")
                if not os.path.exists(sketch_path):
                    score = 1.0
                else:
                    curr_sketch = Image.open(sketch_path).convert("RGB")
                    resample = ensure_resample_mode()
                    curr_sketch = curr_sketch.resize((sizes[idx][0] + 100, sizes[idx][1] + 100), resample)
                    sketch_np = cv2.cvtColor(np.array(curr_sketch), cv2.COLOR_RGB2BGR)
                    contour_item = normalize_filled(sketch_np)
                    try:
                        score = cv2.matchShapes(contour_current, contour_item, 1, 0.0)
                    except Exception:
                        score = 1.0

                scores.append(score)
                extracted_images.append(extracted)
            except Exception as e:
                continue

        if not scores:
            print(f"No candidate matches for item {idx}. Using blank placeholder.")
            selected_scene_items.append(np.zeros((sizes[idx][1] + 100, sizes[idx][0] + 100, 3), dtype=np.uint8))
        else:
            least_score_idx = int(np.argmin(np.array(scores)))
            selected_scene_items.append(extracted_images[least_score_idx])

    print("Composing final images with enhanced blending...")
    final_images = []

    for background_img in selected_background_images:
        dst = cv2.cvtColor(np.array(background_img), cv2.COLOR_RGB2BGR)
        dst = cv2.resize(dst, (512, 512), interpolation=cv2.INTER_LINEAR)

        for ele in range(number_elements):
            src = selected_scene_items[ele]
            if src is None or src.size == 0:
                continue

            if src.ndim == 2:
                src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

            w, h = sizes[ele]
            target_size = (w, h)
            interp = cv2.INTER_AREA if src.shape[1] > w or src.shape[0] > h else cv2.INTER_CUBIC
            src = cv2.resize(src, target_size, interpolation=interp)

            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (7, 7), 3)
            src_mask = mask.astype(np.uint8)

            try:
                src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
                dst_lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
                for i in range(3):
                    src_lab[..., i] = cv2.normalize(
                        src_lab[..., i],
                        None,
                        np.min(dst_lab[..., i]),
                        np.max(dst_lab[..., i]),
                        cv2.NORM_MINMAX
                    )
                src = cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)
            except Exception:
                pass  

            cx = int(desired_locations[ele][0] + w // 2)
            cy = int(desired_locations[ele][1] + h // 2)
            center = (max(0, min(cx, dst.shape[1] - 1)),
                    max(0, min(cy, dst.shape[0] - 1)))

            try:
                dst = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
            except Exception:
                y1 = max(0, center[1] - src.shape[0] // 2)
                x1 = max(0, center[0] - src.shape[1] // 2)
                y2 = min(dst.shape[0], y1 + src.shape[0])
                x2 = min(dst.shape[1], x1 + src.shape[1])
                h_overlap = y2 - y1
                w_overlap = x2 - x1
                if h_overlap > 0 and w_overlap > 0:
                    src_crop = src[0:h_overlap, 0:w_overlap]
                    m = src_mask[0:h_overlap, 0:w_overlap].astype(np.float32) / 255.0
                    m = cv2.GaussianBlur(m, (9, 9), 5)
                    m = np.clip(m[..., None], 0, 1)
                    dst_roi = dst[y1:y2, x1:x2].astype(np.float32)
                    blended = dst_roi * (1 - m) + src_crop.astype(np.float32) * m
                    dst[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        final_images.append(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

    print("Building comparison grid preview...")

    os.makedirs("outputs", exist_ok=True)

    n_display = min(4, len(selected_background_images))
    cols = 4
    rows = n_display

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axs = np.array(axs).reshape(rows, cols)

    for i in range(rows):
        for j in range(cols):
            ax = axs[i, j]
            ax.axis("off")

            if j == 0 and len(selected_scene_items) > 0:
                ax.imshow(cv2.cvtColor(selected_scene_items[0], cv2.COLOR_BGR2RGB))

            elif j == 1 and len(selected_scene_items) > 1:
                ax.imshow(cv2.cvtColor(selected_scene_items[1], cv2.COLOR_BGR2RGB))

            elif j == 2 and i < len(selected_background_images):
                ax.imshow(selected_background_images[i])

            elif j == 3 and i < len(final_images):
                ax.imshow(final_images[i])

    plt.tight_layout()
    grid_path = os.path.join("outputs", "preview_grid_enhanced.jpg")
    fig.savefig(grid_path, dpi=150)
    print(f"Saved comparison grid -> {grid_path}")
    plt.show()



main()

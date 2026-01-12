# infer.py
import sys
from ultralytics import YOLO
import cv2

def run_inference(weights_path, image_path, output_path=None, conf=0.25):
    # Load YOLO model
    model = YOLO(weights_path)

    # Run prediction
    results = model.predict(image_path, conf=conf)

    # Get first result
    result = results[0]

    # Plot predictions on the image
    im_bgr = result.plot()  # plotted image (numpy array BGR)

    # Save output if user provided path
    if output_path:
        cv2.imwrite(output_path, im_bgr)
        print(f"Saved: {output_path}")

    # Show output window
    cv2.imshow("Prediction", im_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------
# CLI usage
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python infer.py <weights_path> <image_path> [output_path] [conf]\n")
        sys.exit(1)

    weights = sys.argv[1]
    image = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else None
    conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25

    run_inference(weights, image, output, conf)
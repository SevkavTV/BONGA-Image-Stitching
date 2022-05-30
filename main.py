from models.camera import Camera
from models.log import Log

if __name__ == "__main__":
    # all images
    log_file_full = Log("flight-logs/fly_info.txt")
    camera = Camera(log_file_full)
    camera.visualize_images("output/all_images.html")

    # set of images used for panorama
    log_file_panorama = Log("flight-logs/fly_info_panorama.txt")
    camera = Camera(log_file_panorama)
    camera.visualize_images("output/panorama_images.html")
    print(camera.images)

from models.camera import Camera
from models.log import Log

if __name__ == "__main__":
    log_file = Log("flight-logs/fly_info.txt")
    camera = Camera(log_file)
    print(camera.images)

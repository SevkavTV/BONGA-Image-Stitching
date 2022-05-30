from models.image import Image
from models.location import Location
from models.rotation import Rotation


class Log:
    def __init__(self, path_to_log: str):
        self.path = path_to_log

    def _retrieve_images_info(self):
        images = []
        with open(self.path, "r") as log_file:
            for row in log_file:
                image = self._create_image_from_row(row)
                if image:
                    images.append(image)

        return images

    def _create_image_from_row(self, row: str):
        if "img_idx" not in row:
            return None

        row_items = row.split()
        location = Location(
            lat=self._get_field_value_from_row_items(row_items, "lat"),
            lot=self._get_field_value_from_row_items(row_items, "lng"),
            alt=self._get_field_value_from_row_items(row_items, "alt_msl"),
        )
        rotation = Rotation(
            yaw=self._get_field_value_from_row_items(row_items, "yaw"),
            roll=self._get_field_value_from_row_items(row_items, "roll"),
            pitch=self._get_field_value_from_row_items(row_items, "pitch"),
        )
        image_id = int(self._get_field_value_from_row_items(row_items, "img_idx"))
        image_path = f"images/{image_id}.JPG"

        return Image(id=image_id, path=image_path, location=location, rotation=rotation)

    def _get_field_value_from_row_items(self, row_items: list, field: str):
        field_id_index = row_items.index(field)
        field_value = row_items[field_id_index + 1].replace(",", ".")
        if field in ["lat", "lng"]:
            field_value = "".join((field_value[:2], ".", field_value[2:]))
        return float(field_value)

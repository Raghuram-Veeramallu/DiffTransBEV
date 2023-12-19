

########## NuScenes constants ############

VALID_DATA_VERSIONS = ['v1.0-mini', 'v1.0-trainval', 'v1.0-test']

# camera sensors
FRONT_CAMERA_SENSOR = 'CAM_FRONT'
FRONT_RIGHT_CAMERA_SENSOR = 'CAM_FRONT_RIGHT'
FRONT_LEFT_CAMERA_SENSOR = 'CAM_FRONT_LEFT'
BACK_CAMERA_SENSOR = 'CAM_BACK'
BACK_RIGHT_CAMERA_SENSOR = 'CAM_BACK_RIGHT'
BACK_LEFT_CAMERA_SENSOR = 'CAM_BACK_LEFT'

# lidar sensor
LIDAR_SENSOR = 'LIDAR_TOP'

# whole camera suite
CAMERA_SENSOR_LIST = [
    FRONT_CAMERA_SENSOR,
    FRONT_RIGHT_CAMERA_SENSOR,
    FRONT_LEFT_CAMERA_SENSOR,
    BACK_CAMERA_SENSOR,
    BACK_RIGHT_CAMERA_SENSOR,
    BACK_LEFT_CAMERA_SENSOR,
]
import carla
import os
import math
import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import threading
import glob
import os
import sys
import time
import cv2
import pygame
import keyboard
import queue

# 6 RGB camera orientations and one spectator camera for game view of the car
CAMERA_POSITIONS = {
    'front': {'z': 1.4, 'x': 0.6, 'y': 0, 'yaw': 0, 'pitch': 8},
    'front_right': {'z': 1.4, 'x': 0.6, 'y': 0.6, 'yaw': 55, 'pitch': 0},
    'front_left': {'z': 1.4, 'x': 0.6, 'y': -0.6, 'yaw': -55, 'pitch': 0},
    'back_left': {'z': 1.6, 'x': -0.6, 'y': -0.6, 'yaw': -110, 'pitch': 0},
    'back_right': {'z': 1.6, 'x': -0.6, 'y': 0.6, 'yaw': 110, 'pitch': 0},
    'back': {'z': 1.6, 'x': -1.3, 'y': 0, 'yaw': 180, 'pitch': 4.75},
    'spectator': {'z': 2.5, 'x': -6, 'y': 0, 'yaw': 0, 'pitch': 0},
}



class CarlaSyncMode(object):
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
    
    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

# initiate the CARLA world
def initiate_world():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    return world

# function to quit monitoring the vehicle
def should_quit():
    return keyboard.is_pressed('q')

# capture the image
def draw_image(image, camera_name, frame):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # BGR
    img_arr = array[:, :, ::-1]  # RGB
    output_dir = f'./camera_feeds/{camera_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f'{frame}.png')
    plt.imsave(filename, img_arr)

def capture_frames():
    actor_list = []
    npc_actors = []

    try:
        world = initiate_world()
        # get the blueprint library
        bp_lib = world.get_blueprint_library()
        # get random spawn points for spawing NPC and main actor car
        spawn_points = world.get_map().get_spawn_points()
        # loading the car as Tesla Model 3
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        # spawn the vehicle
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        # spectate the car from a game view distance
        spectator = world.get_spectator()
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-6, z=2.5)), vehicle.get_transform().rotation)
        spectator.set_transform(transform)
        # add the vehicle to the list of actors
        actor_list.append(vehicle)

        # spawn NPC vehicles and pedestrians (40 vehicles and 20 pedestrians)
        for i in range(40):
            vehicle_bp = random.choice(bp_lib.filter('vehicle'))
            npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            npc_actors.append(npc)
        for i in range(20):
            ped_bp = random.choice(bp_lib.filter('walker'))
            npc = world.try_spawn_actor(ped_bp, random.choice(spawn_points))
            npc_actors.append(npc)
        
        # RGB camera blueprints
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(640))
        camera_bp.set_attribute('image_size_y', str(480))

        # get the camera transformations for the cameras
        camera_transformations = []
        camera_names = []
        for cam_name, vals in CAMERA_POSITIONS.items():
            trans = carla.Location(z=vals['z'], x=vals['x'], y=vals['y'])
            rot = carla.Rotation(yaw=vals['yaw'], pitch=vals['pitch'])
            camera_names.append(cam_name)
            camera_transformations.append(
                carla.Transform(trans, rot)
            )
        
        # place the cameras and attach them to the vehicle
        for transform in camera_transformations:
            camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
            actor_list.append(camera)
        
        # initate pygame
        pygame.init()
        clock = pygame.time.Clock()
        world.tick()
        
        # capture data synchronously
        with CarlaSyncMode(world, *actor_list[1:], fps=20) as sync_mode:
            while True:
                if should_quit():
                    break
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, front, front_right, front_left, back_left, back_right, back, spectator = sync_mode.tick(timeout=2.0)

                frame = str(snapshot.timestamp.frame)
                print(f'Capturing frame: {frame}')

                # Draw the display.
                draw_image(front, 'front', frame)
                draw_image(front_right, 'front_right', frame)
                draw_image(front_left, 'front_left', frame)
                draw_image(back_left, 'back_left', frame)
                draw_image(back_right, 'back_right', frame)
                draw_image(back, 'back', frame)
                draw_image(spectator, 'spectator', frame)

    except Exception as e:
        print(e)
        raise Exception(e)
    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        for actor in npc_actors:
            if actor is not None:
                actor.destroy()
        pygame.quit()


if __name__ == '__main__':
    capture_frames()

import pygame
import os 
import sys
import math
import neat

#CREATING A WINDOW WITH THE TRAKC AND CAR MOVEMENTS
SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
SCREEN  = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))

TRACK = pygame.image.load(os.path.join("Assets","track.png"))

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets","car.png"))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490,820)) #size of the car image
        self.drive_state = False
        self.vel_vector = pygame.math.Vector2(0.8,0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True

    def update(self):
        self.drive()
        self.rotate()
        for radar_angle in (-60,-30,0,30,60):
            self.radar(radar_angle)
        self.collision()
    
    def drive(self):
        if self.drive_state:
            self.rect.center += self.vel_vector * 6
    
    def collision(self):
        length = 40
        # Set collision points on right and left headlights
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                  int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                  int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]
        
        # Die on Collisions
        if SCREEN.get_at(collision_point_right) == pygame.Color(2,105,31,255) \
            or SCREEN.get_at(collision_point_left)  == pygame.Color(2,105,31,255):
            self.alive = False
        
        # Draw collision points

        pygame.draw.circle(SCREEN, (0,255,255,0), collision_point_right,4)
        pygame.draw.circle(SCREEN,(0,255,255,0),collision_point_left,4)
    
    def rotate(self):
        if self.direction == 1:
            self.angle -=self.rotation_vel# Making it rotate clockwise(to the right)
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle +=self.rotation_vel# Making it rotate clockwise(to the right)
            self.vel_vector.rotate_ip(-self.rotation_vel)
        self.image = pygame.transform.rotozoom(self.original_image,self.angle,0.1)
        self.rect = self.image.get_rect(center=self.rect.center)
    
    def radar(self,radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while not SCREEN.get_at((x,y)) == pygame.Color(2,105,31,255) and length < 200:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle))  * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle))  * length)
        
        # Draw radar
        pygame.draw.line(SCREEN, (255,255,255,255), self.rect.center,(x,y),1)
        pygame.draw.circle(SCREEN,(0,255,0,0),(x,y),3)
        

car = pygame.sprite.GroupSingle(Car()) #container that holds a single sprite image: Car

#game loop
def eval_genoms():
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        SCREEN.blit(TRACK,(0,0)) # display 

        # User Input
        user_input = pygame.key.get_pressed()
        if sum(pygame.key.get_pressed()) <= 1:
            car.sprite.drive_state = False
            car.sprite.direction = 0

        # Drive
        if user_input[pygame.K_UP]:
            car.sprite.drive_state = True

        # Steer
        if user_input[pygame.K_RIGHT]:
            car.sprite.direction = 1
        if user_input[pygame.K_LEFT]:
            car.sprite.direction = -1
        
        # Update 
        car.draw(SCREEN)
        car.update()
        pygame.display.update()
eval_genoms()
        
""" import pygame
import os
import sys
import math
import random
import numpy as np

# Constants
SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
TRACK = pygame.image.load(os.path.join("Assets", "track.png"))

# Neural Network
class FeedForwardNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.random.randn(hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.random.randn(output_size)

    def activate(self, x):
        return 1 / (1 + np.exp(-x))  # Sigmoid activation


    def forward(self, x):
        hidden = self.activate(np.dot(x, self.weights1) + self.bias1)
        output = self.activate(np.dot(hidden, self.weights2) + self.bias2)
        return output

# Genetic Algorithm
class GeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden_size, output_size):
        self.population = [FeedForwardNetwork(input_size, hidden_size, output_size) for _ in range(population_size)]
        self.scores = [0] * population_size

    def evolve(self):
        # Sort by fitness scores
        sorted_population = [net for _, net in sorted(zip(self.scores, self.population), key=lambda pair: pair[0], reverse=True)]
        self.population = sorted_population[:len(sorted_population) // 2]  # Elitism

        # Crossover and mutation
        children = []
        for _ in range(len(self.population)):
            parent1, parent2 = random.sample(self.population, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            children.append(child)

        self.population.extend(children)


    def crossover(self, parent1, parent2):
        child = FeedForwardNetwork(len(parent1.weights1), len(parent1.weights1[0]), len(parent1.weights2[0]))
        child.weights1 = (parent1.weights1 + parent2.weights1) / 2
        child.bias1 = (parent1.bias1 + parent2.bias1) / 2
        child.weights2 = (parent1.weights2 + parent2.weights2) / 2
        child.bias2 = (parent1.bias2 + parent2.bias2) / 2
        return child

    def mutate(self, network):
        mutation_rate = 0.1
        if random.random() < mutation_rate:
            network.weights1 += np.random.randn(*network.weights1.shape) * 0.1
            network.bias1 += np.random.randn(*network.bias1.shape) * 0.1
            network.weights2 += np.random.randn(*network.weights2.shape) * 0.1
            network.bias2 += np.random.randn(*network.bias2.shape) * 0.1

# Car Class
class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets", "car.png"))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 820))

        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()

    def drive(self):
        self.rect.center += self.vel_vector * 6

    def collision(self):
        length = 40
        collision_point_right = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)
        ]
        collision_point_left = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)
        ]
        if SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) or \
           SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while not SCREEN.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and length < 200:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))
        self.radars.append([radar_angle, dist])

    def get_data(self):
        inputs = [0] * 5
        for i, radar in enumerate(self.radars):
            inputs[i] = int(radar[1])
        return inputs

# Simulation Loop
def simulate(ga, generations):
    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        
        # Reinitialize cars for the current generation
        cars = [pygame.sprite.GroupSingle(Car()) for _ in ga.population]
        scores = [0] * len(ga.population)

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            SCREEN.blit(TRACK, (0, 0))

            all_dead = True
            for i, car in enumerate(cars):
                if not car.sprite.alive:
                    continue
                
                all_dead = False  # At least one car is alive
                inputs = car.sprite.get_data()
                outputs = ga.population[i].forward(inputs)

                if outputs[0] > 0.7:
                    car.sprite.direction = 1
                if outputs[1] > 0.7:
                    car.sprite.direction = -1
                if outputs[0] <= 0.7 and outputs[1] <= 0.7:
                    car.sprite.direction = 0


                car.update()
                car.draw(SCREEN)
                scores[i] += 1

            pygame.display.update()

            if all_dead:  # Move to the next generation if all cars are dead
                run = False

        # Update fitness scores and evolve the population
        ga.scores = scores
        ga.evolve()

# Main Entry Point
if __name__ == "__main__":
    input_size = 5
    hidden_size = 6
    output_size = 2
    population_size = 20
    generations = 50

    ga = GeneticAlgorithm(population_size, input_size, hidden_size, output_size)
    simulate(ga, generations)
 """
import pygame
import os
import sys
import math
import random
import numpy as np

# Constants
SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
TRACK = pygame.image.load(os.path.join("Assets", "track.png"))

class FeedForwardNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)  # Xavier initialization
        self.bias1 = np.zeros(hidden_size)  # Initialize biases to zero
        self.weights2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.bias2 = np.zeros(output_size)

    def activate(self, x):
        return np.maximum(0, x)  # ReLU activation - better for deep networks

    def forward(self, x):
        self.hidden = self.activate(np.dot(x, self.weights1) + self.bias1)
        return self.activate(np.dot(self.hidden, self.weights2) + self.bias2)

class GeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden_size, output_size):
        self.population = [FeedForwardNetwork(input_size, hidden_size, output_size) for _ in range(population_size)]
        self.scores = [0] * population_size
        self.best_score = 0
        self.generation = 0
        self.checkpoint_positions = [(490, 820), (490, 600), (800, 400), (1000, 200)]  # Add checkpoints
        
    def calculate_fitness(self, car_pos, checkpoints_reached):
        # Calculate distance to next checkpoint
        next_checkpoint = self.checkpoint_positions[checkpoints_reached]
        distance = math.sqrt((car_pos[0] - next_checkpoint[0])**2 + (car_pos[1] - next_checkpoint[1])**2)
        
        # Fitness combines checkpoints reached and distance to next checkpoint
        return (checkpoints_reached * 1000) + (1000 - min(distance, 1000))

    def evolve(self):
        self.generation += 1
        
        # Calculate normalized fitness scores
        fitness_sum = sum(self.scores)
        if fitness_sum > 0:
            normalized_scores = [score / fitness_sum for score in self.scores]
        else:
            normalized_scores = [1.0 / len(self.scores)] * len(self.scores)

        # Tournament selection
        new_population = []
        elite_count = 2  # Keep the best performers
        
        # Elitism - keep the best performers
        sorted_indices = sorted(range(len(self.scores)), key=lambda k: self.scores[k], reverse=True)
        for i in range(elite_count):
            new_population.append(self.population[sorted_indices[i]])
        
        # Tournament selection for the rest
        while len(new_population) < len(self.population):
            tournament_size = 3
            tournament = random.sample(range(len(self.population)), tournament_size)
            winner = max(tournament, key=lambda i: self.scores[i])
            new_population.append(self.crossover(
                self.population[winner],
                self.population[random.choice(tournament)]
            ))

        self.population = new_population
        for network in self.population[elite_count:]:  # Don't mutate the elite
            self.mutate(network)
        
        self.scores = [0] * len(self.population)

    def crossover(self, parent1, parent2):
        child = FeedForwardNetwork(
            len(parent1.weights1), len(parent1.weights1[0]), len(parent1.weights2[0]))
        
        # Uniform crossover with random weight selection
        mask1 = np.random.random(parent1.weights1.shape) < 0.5
        mask2 = np.random.random(parent1.weights2.shape) < 0.5
        
        child.weights1 = np.where(mask1, parent1.weights1, parent2.weights1)
        child.weights2 = np.where(mask2, parent1.weights2, parent2.weights2)
        child.bias1 = (parent1.bias1 + parent2.bias1) / 2
        child.bias2 = (parent1.bias2 + parent2.bias2) / 2
        
        return child

    def mutate(self, network):
        mutation_rate = 0.1
        mutation_strength = 0.2  # Reduced from original
        
        # Weights mutation
        mask1 = np.random.random(network.weights1.shape) < mutation_rate
        mask2 = np.random.random(network.weights2.shape) < mutation_rate
        
        network.weights1 += mask1 * np.random.randn(*network.weights1.shape) * mutation_strength
        network.weights2 += mask2 * np.random.randn(*network.weights2.shape) * mutation_strength
        
        # Bias mutation
        if random.random() < mutation_rate:
            network.bias1 += np.random.randn(*network.bias1.shape) * mutation_strength
            network.bias2 += np.random.randn(*network.bias2.shape) * mutation_strength

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets", "car.png"))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 820))
        
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []
        self.speed = 6
        self.checkpoints_reached = 0
        self.time_alive = 0
        self.last_checkpoint_time = 0
        self.last_position = (490, 820)
        self.stuck_time = 0  # Initialize stuck_time
        
    def update(self):
        self.time_alive += 1
        self.radars.clear()
        self.drive()
        self.rotate()
        
        # Check if car is stuck
        current_pos = self.rect.center
        if math.dist(current_pos, self.last_position) < 1:
            self.stuck_time += 1
            if self.stuck_time > 100:  # If stuck for too long, consider it crashed
                self.alive = False
        else:
            self.stuck_time = 0
        self.last_position = current_pos
        
        # Update radar with five angles
        for radar_angle in (-60, -30, 0, 30, 60):  # Reduced to 5 angles
            self.radar(radar_angle)
        
        self.collision()
        self.check_checkpoint()
        
        # Time limit per checkpoint to prevent infinite loops
        if self.time_alive - self.last_checkpoint_time > 500:
            self.alive = False

    def check_checkpoint(self):
        # Define checkpoint positions and radii
        checkpoints = [
            ((490, 820), 50),  # Start
            ((490, 600), 50),  # Checkpoint 1
            ((800, 400), 50),  # Checkpoint 2
            ((1000, 200), 50)  # Finish
        ]
        
        if self.checkpoints_reached < len(checkpoints):
            checkpoint_pos, radius = checkpoints[self.checkpoints_reached]
            if math.dist(self.rect.center, checkpoint_pos) < radius:
                self.checkpoints_reached += 1
                self.last_checkpoint_time = self.time_alive
                if self.checkpoints_reached == len(checkpoints):
                    print("Finish line reached!")

    def drive(self):
        # Add acceleration and friction
        self.speed = min(10, self.speed)  # Cap maximum speed
        self.rect.center += self.vel_vector * self.speed
    
    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while not SCREEN.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and length < 200:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))
        self.radars.append([radar_angle, dist])

    def get_data(self):
        inputs = [0] * 5
        for i, radar in enumerate(self.radars):
            inputs[i] = int(radar[1])
        return inputs
    
    def collision(self):
        length = 40
        collision_point_right = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)
        ]
        collision_point_left = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)
        ]
        if SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) or \
           SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def get_data(self):
        # Ensure we have exactly 5 radar readings
        inputs = []
        for radar in self.radars:
            inputs.append(radar[1] / 200.0)  # Normalize to 0-1 range
        
        return inputs  # Now returns exactly 5 inputs (one for each radar)


def simulate(ga, generations):
    clock = pygame.time.Clock()
    
    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        
        cars = [pygame.sprite.GroupSingle(Car()) for _ in ga.population]
        
        while any(car.sprite.alive for car in cars):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            SCREEN.blit(TRACK, (0, 0))
            
            # Update and draw each car
            for i, car in enumerate(cars):
                if not car.sprite.alive:
                    continue
                
                inputs = car.sprite.get_data()
                if len(inputs) == 5:  # Ensure we have all 5 inputs before proceeding
                    outputs = ga.population[i].forward(np.array(inputs))
                    
                    # Enhanced control logic
                    if outputs[0] > 0.5:
                        car.sprite.direction = 1
                    elif outputs[1] > 0.5:
                        car.sprite.direction = -1
                    else:
                        car.sprite.direction = 0
                
                car.update()
                car.draw(SCREEN)
                
                # Calculate fitness
                ga.scores[i] = ga.calculate_fitness(
                    car.sprite.rect.center,
                    car.sprite.checkpoints_reached
                )
            
            pygame.display.update()
            clock.tick(60)  # Limit to 60 FPS
        
        # Evolution step
        best_score = max(ga.scores)
        if best_score > ga.best_score:
            ga.best_score = best_score
        
        print(f"Generation {generation + 1} complete")
        print(f"Best score: {ga.best_score}")
        print(f"Average score: {sum(ga.scores) / len(ga.scores)}")
        
        ga.evolve()

if __name__ == "__main__":
    pygame.init()
    
    input_size = 5  # Changed to match the 5 radar inputs
    hidden_size = 8  # Adjusted for the simpler input size
    output_size = 2
    population_size = 30
    generations = 100
    
    ga = GeneticAlgorithm(population_size, input_size, hidden_size, output_size)
    simulate(ga, generations)
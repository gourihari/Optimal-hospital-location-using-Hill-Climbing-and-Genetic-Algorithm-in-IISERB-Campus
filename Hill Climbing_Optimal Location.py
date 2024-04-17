import random
import pandas as pd


class CampusSpace:

    def __init__(self, buildings_df, hospital_starting_point, grid_size):
        self.buildings_df = buildings_df
        self.hospital_starting_point = hospital_starting_point
        self.grid_size = grid_size
        self.hospitals = set()

    def initialize_hospitals(self, num_hospitals):
        self.hospitals = set()
        # Introduce randomness in initial hospital positions (within a small radius)
        max_delta = 0.001  # Adjust this value to control the random offset radius
        for _ in range(num_hospitals):
            random_offset_lat = random.uniform(-max_delta, max_delta)
            random_offset_lon = random.uniform(-max_delta, max_delta)
            initial_position = (
                self.hospital_starting_point[0] + random_offset_lat,
                self.hospital_starting_point[1] + random_offset_lon,
            )
            self.hospitals.add(tuple(initial_position))

    def available_spaces(self):
        candidates = set(
            (row, col) for row in range(self.grid_size[0]) for col in range(self.grid_size[1])
        )

        # Remove all hospitals
        for hospital in self.hospitals:
            candidates.discard(hospital)
        return candidates

    def hill_climb(self, maximum=None, restarts=5, log=False):
        best_hospitals = None
        best_cost = float('inf')

        for _ in range(restarts):
            # Re-initialize hospitals with random positions at each restart
            self.initialize_hospitals(num_hospitals=1)
            if log:
                print(f"Restart #{_+1} - Initial state: cost", self.get_cost(self.hospitals))

            count = 0
            while maximum is None or count < maximum:
                count += 1
                best_neighbors = []
                best_neighbor_cost = None

                # Consider all hospitals to move
                for hospital in self.hospitals:

                    # Consider all neighbors for that hospital
                    for replacement in self.get_neighbors(hospital):

                        # Generate a neighboring set of hospitals
                        neighbor = self.hospitals.copy()
                        neighbor.remove(hospital)
                        neighbor.add(replacement)

                        # Check if neighbor is best so far
                        cost = self.get_cost(neighbor)
                        if best_neighbor_cost is None or cost < best_neighbor_cost:
                            best_neighbor_cost = cost
                            best_neighbors = [neighbor]
                        elif best_neighbor_cost == cost:
                            best_neighbors.append(neighbor)

                # None of the neighbors are better than the current state
                if best_neighbor_cost >= self.get_cost(self.hospitals):
                    break

                # Move to a highest-valued neighbor
                else:
                    if log:
                        print(f"better neighbor: cost {best_neighbor_cost}")
                    self.hospitals = random.choice(best_neighbors)

            # Update best solution if found a better one
            if self.get_cost(self.hospitals) < best_cost:
                best_hospitals = self.hospitals.copy()
                best_cost = self.get_cost(self.hospitals)

        return best_hospitals

    def get_cost(self, hospitals):
        cost = 0
        for _, building in self.buildings_df.iterrows():
            building_coords = (building['y'], building['x'])
            min_distance = min(
                abs(building_coords[0] - hospital[0]) + abs(building_coords[1] - hospital[1])
                for hospital in hospitals
            )
            cost += min_distance
        return cost

    def get_neighbors(self, hospital):
        lat_step = (self.buildings_df['y'].max() - self.buildings_df['y'].min()) / self.grid_size[0]
        lon_step = (self.buildings_df['x'].max() - self.buildings_df['x'].min()) / self.grid_size[1]

        candidates = [
            (hospital[0] - lat_step, hospital[1]),
            (hospital[0] + lat_step, hospital[1]),
            (hospital[0], hospital[1] - lon_step),
            (hospital[0], hospital[1] + lon_step)
        ]

        neighbors = []
        for r, c in candidates:
            if (r, c) in self.hospitals:
                continue
            if self.buildings_df['y'].min() <= r <= self.buildings_df['y'].max() and self.buildings_df['x'].min() <= c <= self.buildings_df['x'].max():
                neighbors.append((r, c))

        return neighbors


dataset_path = "C:/Users/bhava/Desktop/Sem 8/AI/phase 1/latitude and longitude.csv"
campus_data = pd.read_csv(dataset_path)


hospital_starting_point = (23.28540615, 77.27402556)

campus_space = CampusSpace(buildings_df=campus_data, hospital_starting_point=hospital_starting_point, grid_size=(100, 100))

optimal_hospitals = campus_space.hill_climb(log=True)


print("Optimal hospital location:", optimal_hospitals)


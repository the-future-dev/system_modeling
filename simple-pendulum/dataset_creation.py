import numpy as np
import csv
import os

G = 9.81  # gravitational acceleration
RHO = 1.225  # air density

def pendulum_dynamics(theta, omega, t, L, b, A, Cd, F0, omega_drive, torsional_friction=0.03):
    """
    Returns the time derivatives of theta and omega.
    """
    v = L * omega  # tangential velocity
    F_drag = -0.5 * RHO * A * Cd * v * abs(v)  # Drag force (quadratic with velocity)
    F_drive = F0 * np.sin(omega_drive * t)  # Driving force
    
    # Combine all torques
    torque_gravity = -G * L * np.sin(theta)
    torque_drag = F_drag * L
    torque_drive = F_drive * L
    torque_friction = -torsional_friction * omega  # Torsional friction
    torque_total = torque_gravity + torque_drag + torque_drive + torque_friction

    theta_dot = omega
    omega_dot = torque_total/L - b*omega  # Combined angular acceleration

    return theta_dot, omega_dot

def rk4_step(theta, omega, t, L, dt, b, A, Cd, F0, omega_drive):
    k1_theta, k1_omega = pendulum_dynamics(theta, omega, t, L, b, A, Cd, F0, omega_drive)
    k2_theta, k2_omega = pendulum_dynamics(theta + k1_theta*dt/2, omega + k1_omega*dt/2, t+dt/2, L, b, A, Cd, F0, omega_drive)
    k3_theta, k3_omega = pendulum_dynamics(theta + k2_theta*dt/2, omega + k2_omega*dt/2, t+dt/2, L, b, A, Cd, F0, omega_drive)
    k4_theta, k4_omega = pendulum_dynamics(theta + k3_theta*dt, omega + k3_omega*dt, t+dt, L, b, A, Cd, F0, omega_drive)

    theta_new = theta + (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)*dt/6
    omega_new = omega + (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)*dt/6

    return theta_new, omega_new

def simulate_and_save_pendulum_data(theta0=0.1, L=1.0, dt=0.1, T=1000, b=0.01, A=0.01, Cd=0.47, F0=0.0, omega_drive=0.0, sample=1):
    n_steps = int(T/dt)
    theta_data = [theta0]
    omega_data = [0.0]
    time_data = [0]
    
    for i in range(n_steps):
        theta_new, omega_new = rk4_step(theta_data[-1], omega_data[-1], time_data[-1], L, dt, b, A, Cd, F0, omega_drive) 

        theta_data.append(theta_new)
        omega_data.append(omega_new)
        time_data.append(time_data[-1] + dt)

    # Save the data to a CSV file inside the 'dataset' folder
    folder_name = "dataset"
    file_name = f"pendulum_data_{sample}.csv"
    path = os.path.join(folder_name, file_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write constant parameters once at the top
        writer.writerow(["L", "b", "A", "Cd", "F0", "omega_drive"])
        writer.writerow([L, b, A, Cd, F0, omega_drive])
        
        # Add a separator for clarity
        writer.writerow([])
        
        # Now write time series data
        writer.writerow(["Time", "Angular Displacement", "Angular Velocity"])
        for t, th, om in zip(time_data, theta_data, omega_data):
            writer.writerow([t, th, om])

    print(f"Data sample {sample} saved to {path}")

    return theta_data, time_data


def generate_pendulum_dataset(n_samples, time=300):
    all_dataset = []  # this will store both parameters and data for each simulation

    for sample in range(n_samples):
        theta0 = np.random.uniform(-np.pi, np.pi)
        L = np.random.uniform(0.5, 2.0)
        dt = 0.1
        #T = np.random.uniform(500, 1500)
        T = time
        b = np.random.uniform(0.001, 0.1)
        A = np.random.uniform(0.005, 0.02)
        Cd = np.random.uniform(0.2, 0.5)
        F0 = np.random.uniform(0.0, 0.5)
        omega_drive = np.random.uniform(0.0, 2.0*np.pi)

        theta_data, time_data = simulate_and_save_pendulum_data(theta0, L, dt, T, b, A, Cd, F0, omega_drive, sample)
        
        current = {
            'parameters': [L, b, A, Cd, F0, omega_drive],
            'data': {
                'theta_data': theta_data,
                'time_data': time_data
            }
        }
        all_dataset.append(current)

    return all_dataset


def read_pendulum_data(num_datasets, directory="dataset"):
    """
    Function to read a specific number of pendulum data from csv files in a specified directory.
    Each csv file is expected to contain two parts:
    - Parameters: ["L", "b", "A", "Cd", "F0", "omega_drive"]
    - Time series data: ["Time", "Angular Displacement", "Angular Velocity"]
    """
    all_data = []
    dataset_count = 0

    # iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            path = os.path.join(directory, filename)

            with open(path, 'r') as file:
                reader = csv.reader(file)
                
                # Read parameters
                next(reader)  # skip header
                parameters = next(reader)
                parameters = [float(param) for param in parameters]
                
                # Skip separator
                next(reader)
                
                # Read time series data
                next(reader)  # skip header
                time_data = []
                theta_data = []
                omega_data = []
                for row in reader:
                    t, th, om = row
                    time_data.append(float(t))
                    theta_data.append(float(th))
                    omega_data.append(float(om))
                
                current = {
                    'parameters': parameters,
                    'data': {
                        'time_data': time_data,
                        'theta_data': theta_data,
                        'omega_data': omega_data,
                    }
                }
                all_data.append(current)
                
                dataset_count += 1
                print(f"Dataset {dataset_count} read from {path}")

                if dataset_count >= num_datasets:
                    break

    return all_data


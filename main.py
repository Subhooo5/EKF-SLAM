'''
EKF SLAM LOGIC
mu: state estimate, where our best guess of the state is 
sigma: state uncertainty, how uncertain we are of our best guess

2 Steps to the EKF:
- Prediction Update
    - From the control inputs u and some model, how does our state estimate change?
    - Moving only affects the state estimate of the robot
    - Moving affects uncertainty of the system
    - Model noise also affects uncertainty
- Measurement Update
    - From what the robot observes, how do we change our state estimation?
    - We reconcile current uncertainty with uncetainty of measurements


'''

from utils import vehicles, environment
import numpy as np  # type: ignore
import pygame  # type: ignore

# <------ EKF ------>
# ---> ROBOT PARAMETERS
n_state = 3
robot_fov = 3

# ---> LANDMARK PARAMETERS
landmarks = [(4,4),
             (4,8),
             (8,8),
             (12,8),
             (16,8),
             (16,4),
             (12,4),
             (8,4)]
n_landmarks = len(landmarks)

# ---> Noise parameters
R = np.diag([0.002, 0.002, 0.0005])
Q = np.diag((0.003, 0.005))

# ---> EKF Estimation Variables
mu = np.zeros((n_state+2*n_landmarks,1))
sigma = np.zeros((n_state+2*n_landmarks,n_state+2*n_landmarks))

mu[:] = np.nan
np.fill_diagonal(sigma, 100)

# ---> Helpful matrices
Fx = np.block([[np.eye(3), np.zeros((n_state, 2*n_landmarks))]])

# Measurement Simulation functions
def sim_measurements(x, landmarks):
    rx, ry, rtheta = x[0], x[1], x[2]
    zs = [] # List of measurements
    for (lidx,landmark) in enumerate(landmarks): # Iterate over landmarks and indices
        lx,ly = landmark
        dist = np.linalg.norm(np.array([lx-rx,ly-ry])) # distance between robot and landmark
        phi = np.arctan2(ly-ry,lx-rx) - rtheta # angle between robot heading and landmark, relative to robot frame
        phi = np.arctan2(np.sin(phi),np.cos(phi)) # Keep phi bounded, -pi <= phi <= +pi
        if dist < robot_fov: # Only append if observation is within robot field of view
            zs.append((dist,phi,lidx))
    return zs

# ---> EKF Steps
def prediction_update(mu, sigma, u, dt):
    rx, ry, theta = mu[0], mu[1], mu[2]
    v, w = u[0], u[1]
    # Updating state estimate mu with model
    state_model_mat = np.zeros((n_state,1))
    state_model_mat[0] = -(v/w)*np.sin(theta) + (v/w)*np.sin(theta + w*dt) if np.abs(w)>0.01 else v*dt*np.cos(theta)
    state_model_mat[1] = (v/w)*np.cos(theta) - (v/w)*np.cos(theta + w*dt) if np.abs(w)>0.01 else v*dt*np.sin(theta)
    state_model_mat[2] = w*dt
    mu += np.matmul(np.transpose(Fx),state_model_mat)

    # Update state uncertainty with model + noise
    state_jacobian_mat = np.zeros((n_state,n_state))
    state_jacobian_mat[0,2] = -(v/w)*np.cos(theta) + (v/w)*np.cos(theta + w*dt) if np.abs(w)>0.01 else -v*np.sin(theta)*dt
    state_jacobian_mat[1,2] = -(v/w)*np.sin(theta) + (v/w)*np.sin(theta + w*dt) if np.abs(w)>0.01 else v*np.cos(theta)*dt
    G = np.eye(sigma.shape[0]) + np.transpose(Fx).dot(state_jacobian_mat).dot(Fx)
    sigma = G.dot(sigma).dot(np.transpose(G))+np.transpose(Fx).dot(R).dot(Fx)
    return mu, sigma

def measurement_update(mu, sigma, zs):
    rx, ry, theta = mu[0,0], mu[1,0], mu[2,0]
    delta_zs = [np.zeros((2,1)) for lidx in range(n_landmarks)]
    Ks = [np.zeros((mu.shape[0], 2)) for lidx in range(n_landmarks)]
    Hs = [np.zeros((2, mu.shape[0])) for lidx in range(n_landmarks)]
    for z in zs:
        (dist, phi, lidx) = z
        mu_landmark = mu[n_state + lidx*2:n_state + lidx*2+2]
        if np.isnan(mu_landmark[0]):
            mu_landmark[0] = rx + dist*np.cos(phi + theta)
            mu_landmark[1] = ry + dist*np.sin(phi + theta)
            mu[n_state + lidx*2:n_state + lidx*2+2] = mu_landmark
        delta = mu_landmark - np.array([[rx],[ry]])
        q = np.linalg.norm(delta)**2

        dist_est = np.sqrt(q)
        phi_est = np.arctan2(delta[1,0], delta[0,0])-theta; phi_est = np.arctan2(np.sin(phi_est),np.cos(phi_est))
        z_est_arr = np.array([[dist_est],[phi_est]]) # Estimated observation, in numpy array
        z_act_arr = np.array([[dist],[phi]]) # Actual observation in numpy array
        delta_zs[lidx] = z_act_arr-z_est_arr # Difference between actual and estimated observation

        # Helper matrices in computing the measurement update
        Fxj = np.block([[Fx],[np.zeros((2,Fx.shape[1]))]])
        Fxj[n_state:n_state+2,n_state+2*lidx:n_state+2*lidx+2] = np.eye(2)
        H = np.array([[-delta[0,0]/np.sqrt(q),-delta[1,0]/np.sqrt(q),0,delta[0,0]/np.sqrt(q),delta[1,0]/np.sqrt(q)],\
                      [delta[1,0]/q,-delta[0,0]/q,-1,-delta[1,0]/q,+delta[0,0]/q]])
        H = H.dot(Fxj)
        Hs[lidx] = H
        Ks[lidx] = sigma.dot(np.transpose(H)).dot(np.linalg.inv(H.dot(sigma).dot(np.transpose(H)) + Q)) # Add to list of matrices

    mu_offset = np.zeros(mu.shape)
    sigma_factor = np.eye(sigma.shape[0])
    for lidx in range(n_landmarks):
        mu_offset += Ks[lidx].dot(delta_zs[lidx]) # Compute full mu offset
        sigma_factor -= Ks[lidx].dot(Hs[lidx]) # Compute full sigma factor
    mu = mu + mu_offset # Update state estimate
    sigma = sigma_factor.dot(sigma) # Update state uncertainty
    return mu, sigma
# <------ EKF ------>

# <------ PLOTTING ------>

def show_uncertainty_ellipse(env, center, width, angle):
    target_rect = pygame.Rect(center[0]-int(width[0]/2),center[1]-int(width[1]/2),width[0],width[1])
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, env.red, (0, 0, *target_rect.size), 2)
    rotated_surf = pygame.transform.rotate(shape_surf, angle)
    env.get_pygame_surface().blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))

def sigma2transform(sigma_sub):
    # 2x2 mtx, uncertainty in x and y position
    [eigenvals, eigenvecs] = np.linalg.eig(sigma_sub)
    angle = 180*np.arctan2(eigenvecs[1,0], eigenvecs[0,0])/np.pi
    return eigenvals, angle

def show_robot_estimate(mu, sigma, env):
    rx, ry = mu[0], mu[1]
    center = env.position2pixel((rx, ry))
    eigenvals, angle = sigma2transform(sigma[0:2, 0:2])

    width = env.dist2pixellen(eigenvals[0]), env.dist2pixellen(eigenvals[1])
    show_uncertainty_ellipse(env, center, width, angle)

def show_landmark_estimate(mu, sigma, env):
    for lidx in range(n_landmarks): # For each landmark location
        lx,ly,lsigma = mu[n_state+lidx*2], mu[n_state+lidx*2+1], sigma[n_state+lidx*2:n_state+lidx*2+2,n_state+lidx*2:n_state+lidx*2+2]
        if ~np.isnan(lx): # If the landmark has been observed
            p_pixel = env.position2pixel((lx,ly)) # Transform landmark location to pygame surface pixel coordinates
            eigenvals,angle = sigma2transform(lsigma) # Get eigenvalues and rotation angle of covariance of landmark
            if np.max(eigenvals)<15: # Only visualize when the maximum uncertainty is below some threshold
                sigma_pixel = max(env.dist2pixellen(eigenvals[0]),5), max(env.dist2pixellen(eigenvals[1]),5) # Convert eigenvalue units from meters to pixels
                show_uncertainty_ellipse(env,p_pixel,sigma_pixel,angle) # Show the ellipse

def show_landmark_location(landmarks, env):
    for landmark in landmarks:
        lx_pixel, ly_pixel = env.position2pixel(landmark)
        r_pixel = env.dist2pixellen(0.2)
        pygame.gfxdraw.filled_circle(env.get_pygame_surface(), lx_pixel, ly_pixel, r_pixel, (0,255,255))

def show_measurements(x, zs, env):
    rx, ry, theta = x[0], x[1], x[2]
    rx_pix, ry_pix = env.position2pixel((rx,ry)) # Convert robot position units from meters to pixels
    for z in zs: # For each measurement
        dist,phi,lidx = z # Unpack measurement tuple
        lx,ly = rx+dist*np.cos(phi+theta),ry+dist*np.sin(phi+theta) # Set the observed landmark location (lx,ly)
        lx_pix,ly_pix = env.position2pixel((lx,ly)) # Convert observed landmark location units from meters to pixels
        pygame.gfxdraw.line(env.get_pygame_surface(),rx_pix,ry_pix,lx_pix,ly_pix,(155,155,155)) # Draw a line between robot and observed landmark

# <------ PLOTTING ------>


if __name__=='__main__':

    # Initialize pygame
    pygame.init()

    # Initialize robot and time step
    x_init = np.array([1,1,np.pi/2])
    robot = vehicles.DifferentialDrive(x_init)
    dt = 0.01

    # Intialize state estimate
    mu[0:3] = np.expand_dims(x_init, axis=1)
    sigma[0:3, 0:3] = 0.05*np.eye(3)
    sigma[2,2] = 0

    # Initialize and display environment
    env = environment.Environment(map_image_path="./maps/map_blank.png")

    running = True
    u = np.array([0.,0.]) # Controls, forward/backward velocity, angular velocity
    while running:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running = False
            u = robot.update_u(u,event) if event.type==pygame.KEYUP or event.type==pygame.KEYDOWN else u # Update controls based on key states
        robot.move_step(u,dt) # Integrate EOMs forward, i.e., move robot
        # Get measurements
        zs = sim_measurements(robot.get_pose(), landmarks)
        # EKF Slam logic
        mu, sigma = prediction_update(mu, sigma, u, dt)
        mu, sigma = measurement_update(mu, sigma, zs)
        # Show ground truth
        env.show_map() # Re-blit map
        env.show_robot(robot) # Re-blit robot
        show_landmark_location(landmarks, env)
        show_measurements(robot.get_pose(), zs, env)
        # Show EKF estimates
        show_robot_estimate(mu, sigma, env)
        show_landmark_estimate(mu, sigma, env)
        pygame.display.update() # Update display
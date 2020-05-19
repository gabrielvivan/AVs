/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles
  
  // Get normal distribution for x, y, and theta
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  // Sample from these distributions
  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen); 
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  // Get normal distribution for x, y, and theta
  std::default_random_engine gen;
  std::normal_distribution<double> noise_x(0, std_pos[0]);
  std::normal_distribution<double> noise_y(0, std_pos[1]);
  std::normal_distribution<double> noise_theta(0, std_pos[2]);  
  
  // Predict according to motion model
  for (int i = 0; i < num_particles; ++i) {
    double theta = particles[i].theta;

    // Check if yaw rate is zero
    if (fabs(yaw_rate) < 0.0001) {
      particles[i].x += velocity * delta_t * cos(theta);
      particles[i].y += velocity * delta_t * sin(theta);
      
    } else { // or if yaw is not constant
      double vel_yaw = velocity / yaw_rate;
      double yawdt = yaw_rate * delta_t;
      double quantity = theta + yawdt;
      particles[i].x += vel_yaw * (sin(quantity) - sin(theta));
      particles[i].y += vel_yaw * (cos(theta) - cos(quantity));
      particles[i].theta += yawdt;
    }

    // Add gaussian noise measurements
    particles[i].x += noise_x(gen);
    particles[i].y += noise_y(gen);
    particles[i].theta += noise_theta(gen);   
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  int map_id;
  for (unsigned int n = 0; n < observations.size(); ++n) {
    double min_distance = 9999; // Initial value to initialize search   
    for (unsigned int m = 0; m < predicted.size(); ++m) {  
      // Obtain distance between landmarks and predictions
      double distance = dist(observations[n].x, observations[n].y, predicted[m].x, predicted[m].y);
      // Update map_id and min_distance if distance is smaller than current minimum
      if (distance < min_distance) {
        map_id = predicted[m].id;
        min_distance = distance;
      }
    }
    observations[n].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  double x_part, y_part, theta_part, x_map, y_map, calc_dist;
  int id_map;

  for (int i = 0; i < num_particles; ++i) {
    x_part = particles[i].x;
    y_part = particles[i].y;
    theta_part = particles[i].theta;
    particles[i].weight = 1.0;
    
    // Obtain observations within sensor range
    vector<LandmarkObs> predicted;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j){
      x_map = map_landmarks.landmark_list[j].x_f;
      y_map = map_landmarks.landmark_list[j].y_f;
      id_map = map_landmarks.landmark_list[j].id_i;
      calc_dist = dist(x_part, y_part, x_map, y_map);
      if (calc_dist <= sensor_range) {
        predicted.push_back(LandmarkObs{id_map, x_map, y_map});
      }
    }
    
    // Transform observations onto map coordinates
    vector<LandmarkObs> map_observations;
    for (unsigned int j = 0; j < observations.size(); ++j) {
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;
      map_observations.push_back(LandmarkObs{
        observations[j].id,
        x_part + (cos(theta_part) * x_obs) - (sin(theta_part) * y_obs),
        y_part + (sin(theta_part) * x_obs) + (cos(theta_part) * y_obs)});
    }
    
    dataAssociation(predicted, map_observations);
    
    for (unsigned int j = 0; j < map_observations.size(); ++j) {
      double mu_x, mu_y;
      for (unsigned int k = 0; k < predicted.size(); ++k) {
        if (predicted[k].id == map_observations[j].id) {
          mu_x = predicted[k].x;
          mu_y = predicted[k].y;          
        }
      }
      
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      double x_obs = map_observations[j].x;
      double y_obs = map_observations[j].y;
      
      // Calculate normalization term
      double gauss_norm;
      gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

      // Calculate exponent
      double exponent;
      exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
                   + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

      // Calculate weight using normalization terms and exponent
      double weight;
      weight = gauss_norm * exp(-exponent);

      // Update particle weight
      particles[i].weight *= weight;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  vector<double> weights; // Reset weights
  double max_weight = 0.00000001; // Initialize
  
  for (int i = 0; i < num_particles; ++i) {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > max_weight){
      max_weight = particles[i].weight;
    }
  }

  // Wheel sampling method from Sebastian's lesson
  std::uniform_real_distribution<double> dist_double(0.0, max_weight);
  std::uniform_int_distribution<int> dist_int(0, num_particles-1);

  int index = dist_int(gen);
  double beta = 0.0;

  vector<Particle> resampledParticles;
  for (int i = 0; i < num_particles; ++i) {
    beta += dist_double(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }
  particles = resampledParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
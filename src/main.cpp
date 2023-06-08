#include "q_mekf.h"
#include <Eigen/Dense>

#include <iostream>

using namespace Eigen;

int main()
{
    Vector3f sigma_a = {20.78e-3, 20.78e-3, 20.78e-3};
    Vector3f sigma_g = {0.2020*M_PI/180, 0.2020*M_PI/180, 0.2020*M_PI/180};
    Vector3f sigma_m = {3.2e-3, 3.2e-3, 4.1e-3};

    QuaternionMEKF<float, true> mekf(sigma_a, sigma_g, sigma_m);

    Vector3f acc0 = {0, 0, -1};
    Vector3f mag0 = {0.2, 0, 0.4};

    mekf.initialize_from_acc_mag(acc0, mag0);
    Vector4f quat = mekf.quaternion();

    std::cout << "[" << quat[0] << ", " << quat[1] << ", " << quat[2] << ", " << quat[3] << "]" << std::endl;


    Vector3f gyr = {0, 0, 0.05};
    Vector3f acc = {0, 0, -1};
    Vector3f mag = {0.2, 0, 0.4};
    
    int n = 1000;
    while(n--)
    {
        mekf.time_update(gyr, 0.1f);
        mekf.measurement_update(acc, mag);
        // mekf.measurement_update_acc_only(acc);
    }

    quat = mekf.quaternion();
    std::cout << "[" << quat[0] << ", " << quat[1] << ", " << quat[2] << ", " << quat[3] << "]" << std::endl;

    Vector3f bias = mekf.gyroscope_bias();
    std::cout << "[" << bias[0] << ", " << bias[1] << ", " << bias[2] << "]" << std::endl;
}
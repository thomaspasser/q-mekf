#pragma once

#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Geometry>
using Eigen::Matrix;
using Eigen::Quaternion;
using Eigen::Map;

template<typename>
inline constexpr bool dependent_false_v = false;

template <typename T = float, bool with_bias = true>
class QuaternionMEKF
{
    // State dimension
    static constexpr int N = with_bias ? 6 : 3;
    // Measurement dimension
    static const int M = 6;

    typedef Matrix<T, 3, 1> Vector3;
    typedef Matrix<T, 4, 1> Vector4;
    typedef Matrix<T, 6, 1> Vector6;
    typedef Matrix<T, N, N> MatrixN;
    typedef Matrix<T, 3, 3> Matrix3;
    typedef Matrix<T, 4, 4> Matrix4;
    typedef Matrix<T, M, M> MatrixM;
    static constexpr T half = T(1) / T(2);

    public:
        QuaternionMEKF(Vector3 sigma_a, Vector3 sigma_g, Vector3 sigma_m);
        QuaternionMEKF(T sigma_a[3], T sigma_g[3], T sigma_m[3]);
        void initialize_from_acc_mag(Vector3 acc, Vector3 mag);
        void initialize_from_acc_mag(T acc[3], T mag[3]);
        void time_update(Vector3 gyr, T Ts);
        void time_update(T gyr[3], T Ts);
        void measurement_update(Vector3 acc, Vector3 mag);
        void measurement_update(T acc[3], T mag[3]);
        void measurement_update_acc_only(Vector3 acc);
        void measurement_update_acc_only(T acc[3]);
        void measurement_update_mag_only(Vector3 mag);
        void measurement_update_mag_only(T mag[3]);
        Vector4 quaternion();
        MatrixN covariance();
        Vector3 gyroscope_bias();

    private:
        Quaternion<T> qref;

        Vector3 v1ref;
        Vector3 v2ref;

        // State
        Matrix<T, N, 1> x;
        // State covariance
        MatrixN P;

        // Quaternion update matrix
        Matrix4 F;
        // State transition matrix
        MatrixN F_a;

        // Predicted measurement
        Vector6 yhat;
        // Measurement update matrices
        Matrix3 C1, C2;
        // Matrix<T, QMEKF_N_MEAS, QMEKF_DIMENSION> C;
        // Matrix<T, QMEKF_DIMENSION, QMEKF_N_MEAS> K;

        // Constant matrices
        const Matrix3 Racc, Rmag;
        const MatrixM R;
        const MatrixN Q;

        void _measurement_update_partial(const Eigen::Ref<const Vector3>& meas, const Eigen::Ref<const Vector3>& vhat, const Eigen::Ref<const Matrix3>& Rm);
        void _set_transition_matrix(const Eigen::Ref<const Vector3>& gyr, T Ts);
        Matrix3 _skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec);
        Vector3 _accelerometer_measurement_func();
        Vector3 _magnetometer_measurement_func();
        T Tsin(T x);
        T Tcos(T x);

        static constexpr MatrixN _initialize_Q(Vector3 sigma_g);

};
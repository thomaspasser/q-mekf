#include "q_mekf.h"


template <typename T, bool with_bias>
QuaternionMEKF<T, with_bias>::QuaternionMEKF(Vector3 sigma_a, Vector3 sigma_g, Vector3 sigma_m)
    :
      Q( _initialize_Q(sigma_g) ),
      Racc( sigma_a.array().square().matrix().asDiagonal()),
      Rmag( sigma_m.array().square().matrix().asDiagonal()),
      R( (Vector6() << sigma_a, sigma_m).finished().array().square().matrix().asDiagonal() )
{
    qref.setIdentity();    
    x.setZero();

    if constexpr (with_bias)
    {
    // P = MatrixN::Identity();
        P << 1e-6*Matrix3::Identity(), Matrix3::Zero(),
             Matrix3::Zero(), 1e-1*Matrix3::Identity();
    }
    else
    {
        P = 1e-6*Matrix3::Identity();
    }
}

template<typename T, bool with_bias>
QuaternionMEKF<T, with_bias>::QuaternionMEKF(T sigma_a[3], T sigma_g[3], T sigma_m[3]) :
    QuaternionMEKF(Map<Matrix<T, 3, 1>>(sigma_a), Map<Matrix<T, 3, 1>>(sigma_g), Map<Matrix<T, 3, 1>>(sigma_m))
{    
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::initialize_from_acc_mag(Vector3 acc, Vector3 mag)
{
    T anorm = acc.norm();
    v1ref << 0, 0, -anorm;

    Vector3 acc_normalized = acc/anorm;
    Vector3 mag_normalized = mag.normalized();

    Vector3 Rz = -acc_normalized;
    Vector3 Ry = Rz.cross(mag_normalized);
    Vector3 Rx = Ry.cross(Rz);

    Rx.normalize();
    Ry.normalize();
    // Rz is already normalized

    // Construct the rotation matrix
    Matrix3 R;
    R << Rx, Ry, Rz;

    // Eigen can convert it to a quaternion
    qref = R.transpose();

    // Reference magnetic field vector
    v2ref = qref._transformVector(mag);
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::initialize_from_acc_mag(T acc[3], T mag[3])
{
    initialize_from_acc_mag(Map<Matrix<T, 3, 1>>(acc), Map<Matrix<T, 3, 1>>(mag));
}

template <typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::time_update(Vector3 gyr, T Ts)
{
    if constexpr (with_bias)
    {
        _set_transition_matrix(gyr - x.tail(3), Ts);
    }
    else
    {
        _set_transition_matrix(gyr, Ts);
    }

    // Quaternionf.coeffs() get the components in [x,y,z,w] order
    qref = F * qref.coeffs();
    qref.normalize();
    
    // Slice 3x3 block from F
    if constexpr (with_bias)
    {
        F_a << F.block(0, 0, 3, 3), (-Matrix3::Identity()*Ts),
               Matrix3::Zero(), Matrix3::Identity();
    }
    else
    {
        F_a = F.block(0, 0, 3, 3);
    }
    P = F_a * P * F_a.transpose() + Q;
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::time_update(T gyr[3], T Ts)
{
    time_update(Map<Matrix<T, 3, 1>>(gyr), Ts);
}

template <typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update(Vector3 acc, Vector3 mag)
{
    Vector3 v1hat = _accelerometer_measurement_func();
    Vector3 v2hat = _magnetometer_measurement_func();

    C1 = _skew_symmetric_matrix(v1hat);
    C2 = _skew_symmetric_matrix(v2hat);

    Matrix<T, M, N> C;

    if constexpr (with_bias)
    {
        C << C1, Matrix<T, 3, 3>::Zero(),
             C2, Matrix<T, 3, 3>::Zero();
    }
    else
    {
        C << C1, 
            C2;
    }

    yhat << v1hat,
            v2hat;

    Vector6 y;
    y << acc,
         mag;

    Vector6 inno = y - yhat;

    MatrixM s = C * P * C.transpose() + R;

    // K = P * C.T *(s)^-1
    // K * s = P*C.T

    // This is the form 
    // x * A = b
    // Which can be solved with the code below
    Eigen::FullPivLU<MatrixM> lu(s);
    if(lu.isInvertible())
    {
        Matrix<T, N, M> K = P * C.transpose() * lu.inverse();

        x += K * inno;

        // Joseph form of covariance measurement update
        MatrixN temp = MatrixN::Identity() - K * C;
        MatrixN temp2 = temp * P * temp.transpose() + K * R * K.transpose();
        P = temp2;
        // Apply correction to qref
        Quaternion<T> corr(1, half*x(0), half*x(1), half*x(2));
        corr.normalize();
        qref = qref * corr;

        // We only want to reset the quaternion part of the state
        x(0) = 0;
        x(1) = 0;
        x(2) = 0;

    }
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update(T acc[3], T mag[3])
{
    measurement_update(Map<Matrix<T, 3, 1>>(acc), Map<Matrix<T, 3, 1>>(mag));
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::_measurement_update_partial(const Eigen::Ref<const Vector3>& meas, const Eigen::Ref<const Vector3>& vhat, const Eigen::Ref<const Matrix3>& Rm)
{
    C1 = _skew_symmetric_matrix(vhat);

    Matrix<T, 3, N> C;
    if constexpr (with_bias)
    {
        C << C1, Matrix<T, 3, 3>::Zero();
    }
    else
    {
        C = C1;
    }

    Vector3 inno = meas - vhat;

    Matrix3 s = C * P * C.transpose() + Rm;

    // K = P * C.T *(s)^-1
    // K * s = P*C.T

    // This is the form 
    // x * A = b
    // Which can be solved with the code below
    Eigen::FullPivLU<Matrix3> lu(s);
    if(lu.isInvertible())
    {
        Matrix<T, N, 3> K = P * C.transpose() * lu.inverse();

        x += K * inno;

        // Joseph form of covariance measurement update
        MatrixN temp = MatrixN::Identity() - K * C;
        MatrixN temp2 = temp * P * temp.transpose() + K * Racc * K.transpose();
        P = temp2;
        // Apply correction to qref
        Quaternion<T> corr(1, half*x(0), half*x(1), half*x(2));
        corr.normalize();
        qref = qref * corr;

        // We only want to reset the quaternion part of the state
        x(0) = 0;
        x(1) = 0;
        x(2) = 0;

    }
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_acc_only(Vector3 acc)
{
    Vector3 v1hat = _accelerometer_measurement_func();
    _measurement_update_partial(acc, v1hat, Racc);
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_acc_only(T acc[3])
{
    measurement_update_acc_only(Map<Matrix<T, 3, 1>>(acc));
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_mag_only(Vector3 mag)
{
    Vector3 v2hat = _magnetometer_measurement_func();
    _measurement_update_partial(mag, v2hat, Rmag);
}

template<typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::measurement_update_mag_only(T mag[3])
{
    measurement_update_mag_only(Map<Matrix<T, 3, 1>>(mag));
}

template<typename T, bool with_bias>
Matrix<T, 4, 1> QuaternionMEKF<T, with_bias>::quaternion()
{
    return qref.coeffs();
}

template<typename T, bool with_bias>
typename QuaternionMEKF<T, with_bias>::MatrixN QuaternionMEKF<T, with_bias>::covariance()
{
    return P;
}

template<typename T, bool with_bias>
Matrix<T, 3, 1> QuaternionMEKF<T, with_bias>::gyro_bias()
{
    return x.tail(3);
}

template <typename T, bool with_bias>
void QuaternionMEKF<T, with_bias>::_set_transition_matrix(const Eigen::Ref<const Vector3>& gyr, T Ts)
{
    Vector3 delta_theta = gyr*Ts;
    T un = delta_theta.norm();
    if(un == 0) // TODO more robust check
        un = 1e-9;

    Matrix4 Omega;
    Omega << -_skew_symmetric_matrix(delta_theta), delta_theta,
             -delta_theta.transpose(), 0;

    F = Tcos(half*un)*Matrix4::Identity() + Tsin(half*un)/un * Omega;
}

template <typename T, bool with_bias>
Matrix<T, 3, 3> QuaternionMEKF<T, with_bias>::_skew_symmetric_matrix(const Eigen::Ref<const Vector3>& vec)
{
    Matrix3 M;
    M << 0, -vec(2), vec(1),
        vec(2), 0, -vec(0),
        -vec(1), vec(0), 0; 

    return M;
}

template <typename T, bool with_bias>
Matrix<T,3,1> QuaternionMEKF<T, with_bias>::_accelerometer_measurement_func()
{
    // TODO
    // qref.inv() as quaternion rotates [0,0,-a]
    // Probably better to write this out?
    return qref.inverse()._transformVector(v1ref);
}

template <typename T, bool with_bias>
Matrix<T, 3, 1> QuaternionMEKF<T, with_bias>::_magnetometer_measurement_func()
{
    // TODO
    // qref.inv() as quaternion rotates [mx,0,mz]
    // Probably better to write this out?
    return qref.inverse()._transformVector(v2ref);
}

template<typename T, bool with_bias>
T QuaternionMEKF<T, with_bias>::Tsin(T x)
{
    if constexpr (std::is_same_v<T, float>)
        return sinf(x);
    else if constexpr (std::is_same_v<T, double>)
        return sin(x);
    else
        static_assert(dependent_false_v<T>, "Must be float or double.");
}

template<typename T, bool with_bias>
T QuaternionMEKF<T, with_bias>::Tcos(T x)
{
    if constexpr (std::is_same_v<T, float>)
        return cosf(x);
    else if constexpr (std::is_same_v<T, double>)
        return cos(x);
    else
        static_assert(dependent_false_v<T>, "Must be float or double.");
}

template<typename T, bool with_bias>
constexpr typename QuaternionMEKF<T, with_bias>::MatrixN QuaternionMEKF<T, with_bias>::_initialize_Q(Vector3 sigma_g)
{
    if constexpr (with_bias)
    {
        return (Vector6() << sigma_g.array().square().matrix(), 1e-12, 1e-12, 1e-12).finished().asDiagonal();
    }
    else
    {
        return sigma_g.array().square().matrix().asDiagonal();
    }
}


// Explicit instantiation
template class QuaternionMEKF<float, true>;
template class QuaternionMEKF<float, false>;
template class QuaternionMEKF<double, true>;
template class QuaternionMEKF<double, false>;

// ArMath/Quat.h — Quaternions unitaires, SLERP, Shepperd
#pragma once
#include "Mat3d.h"
#include <cmath>

namespace NkMath {

    struct Quat {
        double w, x, y, z;

        Quat() : w(1),x(0),y(0),z(0) {}
        Quat(double w,double x,double y,double z) : w(w),x(x),y(y),z(z) {}

        double Norm2() const { return w*w+x*x+y*y+z*z; }
        double Norm()  const { return std::sqrt(Norm2()); }

        static Quat Identity() { return {1,0,0,0}; }

        Quat Normalized() const {
            double n=Norm(); assert(!nearlyZero(n));
            return {w/n, x/n, y/n, z/n};
        }

        Quat Conjugate() const { return {w,-x,-y,-z}; }

        Quat Inverse() const {
            double n2=Norm2(); assert(!nearlyZero(n2));
            return {w/n2,-x/n2,-y/n2,-z/n2};
        }

        // Produit de Hamilton (composition de rotations, non-commutatif)
        Quat operator*(const Quat& o) const {
            return {
                w*o.w - x*o.x - y*o.y - z*o.z,
                w*o.x + x*o.w + y*o.z - z*o.y,
                w*o.y - x*o.z + y*o.w + z*o.x,
                w*o.z + x*o.y - y*o.x + z*o.w
            };
        }
    };

    // Comparaison antipodal-tolerante
    inline bool ApproxQuat(const Quat& a, const Quat& b, double tol=kEps) {
        double dot = a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
        return std::fabs(dot) > 1.0-tol;
    }

    // Construction depuis axe-angle
    inline Quat FromAxisAngle(const Vec3d& axis, double rad) {
        Vec3d n=axis.Normalized();
        double s=std::sin(rad*0.5), c=std::cos(rad*0.5);
        return {c, n.x*s, n.y*s, n.z*s};
    }

    // Rotation d'un vecteur (formule optimisee sans quaternion intermediaire)
    inline Vec3d Rotate(const Quat& q, const Vec3d& v) {
        Vec3d qv={q.x,q.y,q.z};
        Vec3d uv =Cross(qv,v);
        Vec3d uuv=Cross(qv,uv);
        return v + uv*(2.0*q.w) + uuv*2.0;
    }

    // SLERP : chemin le plus court sur S3, vitesse angulaire constante
    inline Quat Slerp(Quat a, Quat b, double t) {
        double ca = a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
        // correction antipodal
        if (ca < 0.0) { b.w=-b.w; b.x=-b.x; b.y=-b.y; b.z=-b.z; ca=-ca; }
        double k0,k1;
        if (ca > 0.9999) { k0=1-t; k1=t; }
        else {
            double ang=std::acos(ca), sinA=std::sin(ang);
            k0=std::sin((1-t)*ang)/sinA;
            k1=std::sin(t*ang)/sinA;
        }
        return Quat{k0*a.w+k1*b.w, k0*a.x+k1*b.x, k0*a.y+k1*b.y, k0*a.z+k1*b.z}.Normalized();
    }

    // LERP normalise (pour comparaison avec SLERP)
    inline Quat Lerp(Quat a, Quat b, double t) {
        return Quat{a.w+t*(b.w-a.w), a.x+t*(b.x-a.x),
                    a.y+t*(b.y-a.y), a.z+t*(b.z-a.z)}.Normalized();
    }

    // Quat -> Mat3 (Rodrigues)
    inline Mat3d ToMat3(const Quat& q) {
        double xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
        double xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
        double wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
        Mat3d R;
        R(0,0)=1-2*(yy+zz); R(0,1)=2*(xy-wz);   R(0,2)=2*(xz+wy);
        R(1,0)=2*(xy+wz);   R(1,1)=1-2*(xx+zz); R(1,2)=2*(yz-wx);
        R(2,0)=2*(xz-wy);   R(2,1)=2*(yz+wx);   R(2,2)=1-2*(xx+yy);
        return R;
    }

    // Mat3 -> Quat (methode de Shepperd, 4 branches pour stabilite)
    inline Quat FromMat3(const Mat3d& R) {
        double tr=R(0,0)+R(1,1)+R(2,2);
        Quat q;
        if (tr>0) {
            double s=0.5/std::sqrt(tr+1);
            q.w=0.25/s; q.x=(R(2,1)-R(1,2))*s; q.y=(R(0,2)-R(2,0))*s; q.z=(R(1,0)-R(0,1))*s;
        } else if (R(0,0)>R(1,1)&&R(0,0)>R(2,2)) {
            double s=2*std::sqrt(1+R(0,0)-R(1,1)-R(2,2));
            q.w=(R(2,1)-R(1,2))/s; q.x=0.25*s; q.y=(R(0,1)+R(1,0))/s; q.z=(R(0,2)+R(2,0))/s;
        } else if (R(1,1)>R(2,2)) {
            double s=2*std::sqrt(1+R(1,1)-R(0,0)-R(2,2));
            q.w=(R(0,2)-R(2,0))/s; q.x=(R(0,1)+R(1,0))/s; q.y=0.25*s; q.z=(R(1,2)+R(2,1))/s;
        } else {
            double s=2*std::sqrt(1+R(2,2)-R(0,0)-R(1,1));
            q.w=(R(1,0)-R(0,1))/s; q.x=(R(0,2)+R(2,0))/s; q.y=(R(1,2)+R(2,1))/s; q.z=0.25*s;
        }
        return q.Normalized();
    }

} // namespace NkMath

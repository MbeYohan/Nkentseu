// ArMath/Vec3d.h — Vecteur 3D double precision
#pragma once
#include <cmath>
#include <cassert>
#include <cstddef>
#include "Vec2d.h"

namespace NkMath {

    struct Vec3d {
        double x, y, z;

        Vec3d()                          : x(0.0), y(0.0), z(0.0) {}
        Vec3d(double a, double b, double c): x(a),   y(b),   z(c)   {}
        Vec3d(const Vec2d& v, double c)  : x(v.x),  y(v.y), z(c)   {}
        explicit Vec3d(double s)         : x(s),    y(s),   z(s)   {}

        double& operator[](int i) {
            assert(i >= 0 && i < 3 && "Vec3d: index hors bornes");
            return (&x)[i];
        }
        const double& operator[](int i) const {
            assert(i >= 0 && i < 3 && "Vec3d: index hors bornes");
            return (&x)[i];
        }

        Vec3d operator+(const Vec3d& o) const { return {x+o.x, y+o.y, z+o.z}; }
        Vec3d operator-(const Vec3d& o) const { return {x-o.x, y-o.y, z-o.z}; }
        Vec3d operator*(double s)       const { return {x*s,   y*s,   z*s};   }
        Vec3d operator/(double s)       const { assert(!nearlyZero(s)); return {x/s,y/s,z/s}; }
        Vec3d operator-()               const { return {-x, -y, -z}; }

        Vec3d& operator+=(const Vec3d& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
        Vec3d& operator-=(const Vec3d& o) { x-=o.x; y-=o.y; z-=o.z; return *this; }
        Vec3d& operator*=(double s)       { x*=s;   y*=s;   z*=s;   return *this; }

        double Norm2() const { return x*x + y*y + z*z; }
        double Norm()  const { return std::sqrt(Norm2()); }

        Vec3d Normalized() const {
            double n = Norm();
            if (nearlyZero(n)) return Vec3d(0.0);
            return {x/n, y/n, z/n};
        }
        bool IsNormalized(double tol = kEps) const {
            return approxEq(Norm2(), 1.0, tol);
        }

        void Print() const { printf("Vec3d(%.6f, %.6f, %.6f)\n", x, y, z); }
    };

    inline double Dot(const Vec3d& a, const Vec3d& b) {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    // Produit vectoriel 3D : perpendiculaire a a et b, norme = |a||b|sin(theta)
    // Regle main droite : X x Y = Z
    inline Vec3d Cross(const Vec3d& a, const Vec3d& b) {
        return {
            a.y*b.z - a.z*b.y,
            a.z*b.x - a.x*b.z,
            a.x*b.y - a.y*b.x
        };
    }

    inline Vec3d Lerp(const Vec3d& a, const Vec3d& b, double t) {
        return { a.x+t*(b.x-a.x), a.y+t*(b.y-a.y), a.z+t*(b.z-a.z) };
    }

    inline Vec3d operator*(double s, const Vec3d& v) { return v * s; }

    inline bool ApproxVec(const Vec3d& a, const Vec3d& b, double tol = kEps) {
        return approxEq(a.x,b.x,tol) && approxEq(a.y,b.y,tol) && approxEq(a.z,b.z,tol);
    }

    // Projection de a sur b
    inline Vec3d Project(const Vec3d& a, const Vec3d& b) {
        double b2 = b.Norm2();
        assert(!nearlyZero(b2) && "Project: b est nul");
        return b * (Dot(a,b) / b2);
    }

    // Rejection : composante de a orthogonale a b
    inline Vec3d Reject(const Vec3d& a, const Vec3d& b) {
        return a - Project(a, b);
    }

    static_assert(sizeof(Vec3d) == 24,  "Vec3d doit faire 24 octets");
    static_assert(offsetof(Vec3d, x) == 0,  "x doit etre premier");
    static_assert(offsetof(Vec3d, y) == 8,  "y a l'offset 8");
    static_assert(offsetof(Vec3d, z) == 16, "z a l'offset 16");

} // namespace NkMath

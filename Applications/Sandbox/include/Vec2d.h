// ArMath/Vec2d.h — Vecteur 2D double precision
#pragma once
#include <cmath>
#include <cassert>
#include <cstddef>
#include "Float.h"

namespace NkMath {

    struct Vec2d {
        double x, y;

        Vec2d()                     : x(0.0), y(0.0) {}
        Vec2d(double a, double b)   : x(a),   y(b)   {}
        explicit Vec2d(double s)    : x(s),   y(s)   {}

        // Acces par indice (contiguïte garantie C++17)
        double& operator[](int i) {
            assert(i >= 0 && i < 2 && "Vec2d: index hors bornes");
            return (&x)[i];
        }
        const double& operator[](int i) const {
            assert(i >= 0 && i < 2 && "Vec2d: index hors bornes");
            return (&x)[i];
        }

        Vec2d operator+(const Vec2d& o) const { return {x+o.x, y+o.y}; }
        Vec2d operator-(const Vec2d& o) const { return {x-o.x, y-o.y}; }
        Vec2d operator*(double s)       const { return {x*s,   y*s};   }
        Vec2d operator/(double s)       const { assert(!nearlyZero(s)); return {x/s, y/s}; }
        Vec2d operator-()               const { return {-x, -y}; }

        Vec2d& operator+=(const Vec2d& o) { x+=o.x; y+=o.y; return *this; }
        Vec2d& operator-=(const Vec2d& o) { x-=o.x; y-=o.y; return *this; }
        Vec2d& operator*=(double s)       { x*=s;   y*=s;   return *this; }

        double LengthSq() const { return x*x + y*y; }
        double Length()   const { return std::sqrt(LengthSq()); }
        // Alias exiges par certains TP
        double Norm2() const { return LengthSq(); }
        double Norm()  const { return Length(); }

        Vec2d Normalized() const {
            double len = Length();
            if (nearlyZero(len)) return Vec2d(0.0);
            return {x/len, y/len};
        }
        bool IsNormalized(double tol = kEps) const {
            return approxEq(LengthSq(), 1.0, tol);
        }

        void Print() const { printf("Vec2d(%.6f, %.6f)\n", x, y); }
    };

    inline double Dot(const Vec2d& a, const Vec2d& b) {
        return a.x*b.x + a.y*b.y;
    }

    // Produit 2D : composante z du cross 3D
    inline double Cross2D(const Vec2d& a, const Vec2d& b) {
        return a.x*b.y - a.y*b.x;
    }

    inline Vec2d Lerp(const Vec2d& a, const Vec2d& b, double t) {
        return { a.x + t*(b.x-a.x), a.y + t*(b.y-a.y) };
    }

    inline Vec2d operator*(double s, const Vec2d& v) { return v * s; }

    static_assert(sizeof(Vec2d) == 16,            "Vec2d doit faire 16 octets");
    static_assert(offsetof(Vec2d, x) == 0,        "x doit etre en premier");
    static_assert(offsetof(Vec2d, y) == 8,        "y doit etre a l'offset 8");

} // namespace NkMath

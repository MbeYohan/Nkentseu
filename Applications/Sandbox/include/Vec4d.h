// ArMath/Vec4d.h — Coordonnees homogenes 4D
#pragma once
#include "Vec3d.h"

namespace NkMath {

    struct Vec4d {
        double x, y, z, w;

        Vec4d()                                      : x(0),y(0),z(0),w(0) {}
        Vec4d(double a,double b,double c,double d)   : x(a),y(b),z(c),w(d) {}
        Vec4d(const Vec3d& v, double d)              : x(v.x),y(v.y),z(v.z),w(d) {}

        // w=1 -> point (translation appliquee)
        // w=0 -> direction (translation ignoree)
        Vec3d ToVec3() const {
            assert(!nearlyZero(w) && "ToVec3: w nul (c'est une direction)");
            return {x/w, y/w, z/w};
        }

        double& operator[](int i) { assert(i<4); return (&x)[i]; }
        const double& operator[](int i) const { assert(i<4); return (&x)[i]; }

        Vec4d operator*(double s) const { return {x*s,y*s,z*s,w*s}; }
        Vec4d operator+(const Vec4d& o) const { return {x+o.x,y+o.y,z+o.z,w+o.w}; }
    };

    // Projection perspective simple (z_cam deja ajoute)
    inline Vec2d ProjectPoint(const Vec4d& p,
                               double fx=500, double fy=500,
                               double cx=256, double cy=256) {
        return { fx*(p.x/p.z)+cx, fy*(p.y/p.z)+cy };
    }

    // NDC -> coordonnees ecran
    inline Vec3d ProjectToScreen(const Vec4d& v, int W, int H) {
        double nx = v.x / v.w;
        double ny = v.y / v.w;
        int px = (int)((nx*0.5 + 0.5) * W);
        int py = (int)((1.0 - (ny*0.5 + 0.5)) * H);
        return {(double)px, (double)py, 0.0};
    }

    static_assert(sizeof(Vec4d) == 32, "Vec4d doit faire 32 octets");

} // namespace NkMath

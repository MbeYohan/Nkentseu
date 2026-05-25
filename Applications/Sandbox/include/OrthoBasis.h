// ArMath/OrthoBasis.h — Base orthonormale via Gram-Schmidt
#pragma once
#include "Vec3d.h"

namespace NkMath {

    struct OrthoBasis {
        Vec3d u, v, w;   // vecteurs orthogonaux et unitaires
    };

    // Gram-Schmidt : transforme 3 vecteurs quelconques en base orthonormale
    // Etape 1 : normaliser a -> u
    // Etape 2 : oter la projection de b sur u -> normaliser -> v
    // Etape 3 : oter les projections de c sur u et v -> normaliser -> w
    inline OrthoBasis GramSchmidt(Vec3d a, Vec3d b, Vec3d c) {
        Vec3d u = a.Normalized();
        assert(!nearlyZero(u.Norm()) && "GramSchmidt: premier vecteur nul");

        Vec3d v = (b - Project(b, u)).Normalized();
        assert(!nearlyZero(v.Norm()) && "GramSchmidt: vecteurs colineaires");

        Vec3d w = (c - Project(c,u) - Project(c,v)).Normalized();
        assert(!nearlyZero(w.Norm()) && "GramSchmidt: vecteurs coplanaires");

        return {u, v, w};
    }

} // namespace NkMath

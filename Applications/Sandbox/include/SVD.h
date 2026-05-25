// ArMath/SVD.h — Decomposition SVD 3x3 via Jacobi
#pragma once
#include "Mat4d.h"
#include <cmath>

namespace NkMath {

    struct SVD3x3 {
        Mat3d U;      // matrice orthogonale gauche (m x m)
        Vec3d sigma;  // valeurs singulieres (triees decroissant)
        Mat3d V;      // matrice orthogonale droite (n x n)

        // Residu de reconstruction ||A - U*S*Vt||
        double residual(const Mat3d& A) const {
            Mat3d S{};
            S(0,0)=sigma.x; S(1,1)=sigma.y; S(2,2)=sigma.z;
            Mat3d Rec = U * S * V.Transposed();
            double err=0;
            for(int i=0;i<3;i++) for(int j=0;j<3;j++){
                double d=Rec(i,j)-A(i,j); err+=d*d;
            }
            return std::sqrt(err);
        }

        // Pseudo-inverse A+ = V * S+ * Ut
        Mat3d pseudoInverse(double thr=1e-10) const {
            Mat3d Sp{};
            if(sigma.x>thr) Sp(0,0)=1.0/sigma.x;
            if(sigma.y>thr) Sp(1,1)=1.0/sigma.y;
            if(sigma.z>thr) Sp(2,2)=1.0/sigma.z;
            return V * Sp * U.Transposed();
        }

        // Vecteur du noyau (derniere colonne de V)
        Vec3d nullSpaceVector() const { return V.col(2); }

        // Rotation pure : R = U * Vt (decomposition polaire)
        Mat3d rotationPart() const {
            Mat3d R = U * V.Transposed();
            if (R.Det() < 0) {
                Mat3d Uf=U;
                for(int i=0;i<3;i++) Uf(i,2)*=-1;
                R = Uf * V.Transposed();
            }
            return R;
        }
    };

    // Decomposition propre d'une matrice symetrique 3x3 (Jacobi)
    inline void jacobiEigen3x3(const Mat3d& A, Mat3d& eigVec, Vec3d& eigVal) {
        Mat3d M=A;
        eigVec=Mat3d::Identity();

        for(int iter=0; iter<50; iter++){
            // plus grand element hors-diagonal
            int p=0, q=1;
            double mx=std::fabs(M(0,1));
            if(std::fabs(M(0,2))>mx){p=0;q=2;mx=std::fabs(M(0,2));}
            if(std::fabs(M(1,2))>mx){p=1;q=2;mx=std::fabs(M(1,2));}
            if(mx<1e-12) break;

            double th=0.5*(M(q,q)-M(p,p))/M(p,q);
            double t=(th>=0) ? 1.0/(th+std::sqrt(th*th+1))
                             : -1.0/(-th+std::sqrt(th*th+1));
            double c=1.0/std::sqrt(t*t+1), s=t*c;

            // rotation de Givens
            for(int k=0;k<3;k++){
                double mkp=M(k,p), mkq=M(k,q);
                M(k,p)=c*mkp-s*mkq; M(k,q)=s*mkp+c*mkq;
            }
            for(int k=0;k<3;k++){
                double mpk=M(p,k), mqk=M(q,k);
                M(p,k)=c*mpk-s*mqk; M(q,k)=s*mpk+c*mqk;
            }
            for(int k=0;k<3;k++){
                double vkp=eigVec(k,p), vkq=eigVec(k,q);
                eigVec(k,p)=c*vkp-s*vkq; eigVec(k,q)=s*vkp+c*vkq;
            }
        }
        eigVal={M(0,0),M(1,1),M(2,2)};

        // tri decroissant
        for(int i=0;i<3;i++) for(int j=i+1;j<3;j++){
            if(eigVal[j]>eigVal[i]){
                std::swap(eigVal[i],eigVal[j]);
                for(int k=0;k<3;k++) std::swap(eigVec(k,i),eigVec(k,j));
            }
        }
    }

    // SVD 3x3 via decomposition propre de A^T * A
    inline SVD3x3 svd3x3(const Mat3d& A) {
        Mat3d AtA = A.Transposed() * A;
        Mat3d V; Vec3d D;
        jacobiEigen3x3(AtA, V, D);

        Vec3d sig{
            std::sqrt(std::max(0.0,D.x)),
            std::sqrt(std::max(0.0,D.y)),
            std::sqrt(std::max(0.0,D.z))
        };

        Mat3d U;
        for(int i=0;i<3;i++){
            if(sig[i]>kEps){
                Vec3d vi=V.col(i);
                Vec3d ui=(A*vi)*(1.0/sig[i]);
                U.setCol(i, ui.Normalized());
            }
        }
        return {U, sig, V};
    }

    // Decomposition polaire : extrait la rotation pure d'une matrice
    inline Mat3d polarRotation(const Mat3d& A){
        return svd3x3(A).rotationPart();
    }

} // namespace NkMath

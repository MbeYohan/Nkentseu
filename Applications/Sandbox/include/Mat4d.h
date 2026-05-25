// ArMath/Mat4d.h — Matrice 4x4 column-major (convention OpenGL)
#pragma once
#include "Vec4d.h"
#include "Mat3d.h"
#include <algorithm>
#include <cmath>

namespace NkMath {

    struct Mat4d {
        double data[16];  // column-major : data[col*4+row]

        Mat4d() { std::fill(data, data+16, 0.0); }

        double& operator()(int row, int col) {
            assert(row>=0&&row<4&&col>=0&&col<4);
            return data[col*4+row];
        }
        const double& operator()(int row, int col) const {
            assert(row>=0&&row<4&&col>=0&&col<4);
            return data[col*4+row];
        }

        static Mat4d Identity() {
            Mat4d m; m(0,0)=m(1,1)=m(2,2)=m(3,3)=1.0; return m;
        }

        Mat4d operator*(const Mat4d& o) const {
            Mat4d r;
            for(int i=0;i<4;i++) for(int j=0;j<4;j++){
                double s=0; for(int k=0;k<4;k++) s+=(*this)(i,k)*o(k,j);
                r(i,j)=s;
            }
            return r;
        }

        Vec4d operator*(const Vec4d& v) const {
            return {
                (*this)(0,0)*v.x+(*this)(0,1)*v.y+(*this)(0,2)*v.z+(*this)(0,3)*v.w,
                (*this)(1,0)*v.x+(*this)(1,1)*v.y+(*this)(1,2)*v.z+(*this)(1,3)*v.w,
                (*this)(2,0)*v.x+(*this)(2,1)*v.y+(*this)(2,2)*v.z+(*this)(2,3)*v.w,
                (*this)(3,0)*v.x+(*this)(3,1)*v.y+(*this)(3,2)*v.z+(*this)(3,3)*v.w
            };
        }

        Mat4d Transposed() const {
            Mat4d t;
            for(int i=0;i<4;i++) for(int j=0;j<4;j++) t(i,j)=(*this)(j,i);
            return t;
        }

        const double* DataPtr() const { return data; }
        void ToFloat(float out[16]) const { for(int i=0;i<16;i++) out[i]=(float)data[i]; }

        // Rodrigues 4x4
        static Mat4d RotateAxis(const Vec3d& axis, double rad) {
            Vec3d n=axis.Normalized();
            double c=std::cos(rad), s=std::sin(rad), t=1-c;
            Mat4d R=Mat4d::Identity();
            R(0,0)=t*n.x*n.x+c;     R(0,1)=t*n.x*n.y-s*n.z; R(0,2)=t*n.x*n.z+s*n.y;
            R(1,0)=t*n.x*n.y+s*n.z; R(1,1)=t*n.y*n.y+c;     R(1,2)=t*n.y*n.z-s*n.x;
            R(2,0)=t*n.x*n.z-s*n.y; R(2,1)=t*n.y*n.z+s*n.x; R(2,2)=t*n.z*n.z+c;
            return R;
        }
    };

    // Gauss-Jordan avec pivot partiel
    inline bool Inverse(const Mat4d& m, Mat4d& out) {
        double a[4][8];
        for(int r=0;r<4;r++) for(int c=0;c<4;c++){
            a[r][c]=m(r,c); a[r][c+4]=(r==c)?1.0:0.0;
        }
        for(int col=0;col<4;col++){
            int piv=col;
            for(int r=col+1;r<4;r++) if(std::fabs(a[r][col])>std::fabs(a[piv][col])) piv=r;
            if(piv!=col) for(int c=0;c<8;c++) std::swap(a[col][c],a[piv][c]);
            if(nearlyZero(a[col][col])) return false;
            double inv=1.0/a[col][col];
            for(int c=0;c<8;c++) a[col][c]*=inv;
            for(int r=0;r<4;r++){
                if(r==col) continue;
                double f=a[r][col];
                for(int c=0;c<8;c++) a[r][c]-=f*a[col][c];
            }
        }
        for(int r=0;r<4;r++) for(int c=0;c<4;c++) out(r,c)=a[r][c+4];
        return true;
    }

    inline bool ApproxMat(const Mat4d& A, const Mat4d& B, double tol=kEps) {
        for(int i=0;i<4;i++) for(int j=0;j<4;j++)
            if(!approxEq(A(i,j),B(i,j),tol)) return false;
        return true;
    }

    // =========================================================
    // Matrice View : LookAt
    // Camera en 'eye', regardant 'target', avec 'worldUp' comme haut
    // =========================================================
    inline Mat4d LookAt(const Vec3d& eye, const Vec3d& target, const Vec3d& worldUp) {
        Vec3d fwd   = (target-eye).Normalized();
        Vec3d right = Cross(fwd, worldUp).Normalized();
        Vec3d up    = Cross(right, fwd);

        Mat4d V=Mat4d::Identity();
        V(0,0)=right.x;  V(0,1)=right.y;  V(0,2)=right.z;
        V(1,0)=up.x;     V(1,1)=up.y;     V(1,2)=up.z;
        V(2,0)=-fwd.x;   V(2,1)=-fwd.y;   V(2,2)=-fwd.z;
        V(0,3)=-Dot(right,eye);
        V(1,3)=-Dot(up,eye);
        V(2,3)= Dot(fwd,eye);
        return V;
    }

    // =========================================================
    // Projection perspective (fovY en degres)
    // =========================================================
    inline Mat4d Perspective(double fovYdeg, double aspect, double zNear, double zFar) {
        double fovRad  = fovYdeg * 3.14159265358979323846 / 180.0;
        double halfTan = std::tan(fovRad * 0.5);
        Mat4d P;
        P(0,0) = 1.0 / (aspect * halfTan);
        P(1,1) = 1.0 / halfTan;
        P(2,2) = -(zFar+zNear)/(zFar-zNear);
        P(2,3) = -(2.0*zFar*zNear)/(zFar-zNear);
        P(3,2) = -1.0;
        return P;
    }

    // =========================================================
    // Transformations 3D
    // =========================================================
    inline Mat4d Translate(const Vec3d& t){
        Mat4d M=Mat4d::Identity(); M(0,3)=t.x; M(1,3)=t.y; M(2,3)=t.z; return M;
    }
    inline Mat4d Scale(const Vec3d& s){
        Mat4d M=Mat4d::Identity(); M(0,0)=s.x; M(1,1)=s.y; M(2,2)=s.z; return M;
    }

    // TRS 3D : T * R * S avec angles Euler (XYZ)
    inline Mat4d TRS(const Vec3d& t, const Vec3d& euler, const Vec3d& s) {
        Mat4d Rx=Mat4d::RotateAxis({1,0,0},euler.x);
        Mat4d Ry=Mat4d::RotateAxis({0,1,0},euler.y);
        Mat4d Rz=Mat4d::RotateAxis({0,0,1},euler.z);
        return Translate(t) * Rz * Ry * Rx * Scale(s);
    }

    // Decomposition TRS 3D
    inline void DecomposeTRS(const Mat4d& M, Vec3d& outT, Vec3d& outR, Vec3d& outS) {
        outT = {M(0,3), M(1,3), M(2,3)};
        Vec3d c0{M(0,0),M(1,0),M(2,0)};
        Vec3d c1{M(0,1),M(1,1),M(2,1)};
        Vec3d c2{M(0,2),M(1,2),M(2,2)};
        outS = {c0.Norm(), c1.Norm(), c2.Norm()};
        // angles Euler depuis matrice rotation normalisee
        double r00=M(0,0)/outS.x, r10=M(1,0)/outS.x, r20=M(2,0)/outS.x;
        double r21=M(2,1)/outS.y, r22=M(2,2)/outS.z;
        outR.x = std::atan2(r21, r22);
        outR.y = std::atan2(-r20, std::sqrt(r21*r21+r22*r22));
        outR.z = std::atan2(r10, r00);
    }

    // Construire Mat4d depuis R3x3 + translation (resultat solvePnP)
    inline Mat4d FromRT(const Mat3d& R, const Vec3d& t) {
        Mat4d M=Mat4d::Identity();
        for(int i=0;i<3;i++) for(int j=0;j<3;j++) M(i,j)=R(i,j);
        M(0,3)=t.x; M(1,3)=t.y; M(2,3)=t.z;
        return M;
    }

} // namespace NkMath

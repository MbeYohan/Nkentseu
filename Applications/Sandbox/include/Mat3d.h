// ArMath/Mat3d.h — Matrice 3x3 column-major
#pragma once
#include "Vec3d.h"
#include <algorithm>
#include <cmath>

namespace NkMath {

    struct Mat3d {
        double data[9];  // column-major : data[col*3+row]

        Mat3d() { std::fill(data, data+9, 0.0); }

        double& operator()(int row, int col) {
            assert(row>=0&&row<3&&col>=0&&col<3);
            return data[col*3+row];
        }
        const double& operator()(int row, int col) const {
            assert(row>=0&&row<3&&col>=0&&col<3);
            return data[col*3+row];
        }

        static Mat3d Identity() {
            Mat3d m; m(0,0)=m(1,1)=m(2,2)=1.0; return m;
        }

        Mat3d operator*(const Mat3d& o) const {
            Mat3d r;
            for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
                double s=0; for (int k=0;k<3;k++) s+=(*this)(i,k)*o(k,j);
                r(i,j)=s;
            }
            return r;
        }

        Mat3d operator-(const Mat3d& o) const {
            Mat3d r; for(int i=0;i<9;i++) r.data[i]=data[i]-o.data[i]; return r;
        }

        Vec3d operator*(const Vec3d& v) const {
            return {
                (*this)(0,0)*v.x+(*this)(0,1)*v.y+(*this)(0,2)*v.z,
                (*this)(1,0)*v.x+(*this)(1,1)*v.y+(*this)(1,2)*v.z,
                (*this)(2,0)*v.x+(*this)(2,1)*v.y+(*this)(2,2)*v.z
            };
        }

        Mat3d Transposed() const {
            Mat3d t;
            for(int i=0;i<3;i++) for(int j=0;j<3;j++) t(i,j)=(*this)(j,i);
            return t;
        }

        double Det() const {
            return (*this)(0,0)*((*this)(1,1)*(*this)(2,2)-(*this)(1,2)*(*this)(2,1))
                  -(*this)(0,1)*((*this)(1,0)*(*this)(2,2)-(*this)(1,2)*(*this)(2,0))
                  +(*this)(0,2)*((*this)(1,0)*(*this)(2,1)-(*this)(1,1)*(*this)(2,0));
        }

        double norm() const {
            double s=0;
            for(int i=0;i<3;i++) for(int j=0;j<3;j++) s+=(*this)(i,j)*(*this)(i,j);
            return std::sqrt(s);
        }

        // Acces ligne/colonne
        Vec3d col(int c) const { return {(*this)(0,c),(*this)(1,c),(*this)(2,c)}; }
        Vec3d row(int r) const { return {(*this)(r,0),(*this)(r,1),(*this)(r,2)}; }
        void setCol(int c,const Vec3d& v){(*this)(0,c)=v.x;(*this)(1,c)=v.y;(*this)(2,c)=v.z;}
        void setRow(int r,const Vec3d& v){(*this)(r,0)=v.x;(*this)(r,1)=v.y;(*this)(r,2)=v.z;}

        const double* DataPtr() const { return data; }
        void ToFloat(float out[9]) const { for(int i=0;i<9;i++) out[i]=(float)data[i]; }

        // Rotation via Rodrigues (3D -> dans une Mat3d)
        static Mat3d RotateAxis(const Vec3d& axis, double rad) {
            Vec3d n=axis.Normalized();
            double c=std::cos(rad), s=std::sin(rad), t=1-c;
            Mat3d R=Mat3d::Identity();
            R(0,0)=t*n.x*n.x+c;       R(0,1)=t*n.x*n.y-s*n.z; R(0,2)=t*n.x*n.z+s*n.y;
            R(1,0)=t*n.x*n.y+s*n.z;   R(1,1)=t*n.y*n.y+c;     R(1,2)=t*n.y*n.z-s*n.x;
            R(2,0)=t*n.x*n.z-s*n.y;   R(2,1)=t*n.y*n.z+s*n.x; R(2,2)=t*n.z*n.z+c;
            return R;
        }
    };

    // Inverse 3x3 par Gauss-Jordan
    inline bool Inverse(const Mat3d& m, Mat3d& out) {
        double a[3][6];
        for(int r=0;r<3;r++) for(int c=0;c<3;c++) {
            a[r][c]=m(r,c); a[r][c+3]=(r==c)?1.0:0.0;
        }
        for(int col=0;col<3;col++){
            int piv=col;
            for(int r=col+1;r<3;r++) if(std::fabs(a[r][col])>std::fabs(a[piv][col])) piv=r;
            if(piv!=col) for(int c=0;c<6;c++) std::swap(a[col][c],a[piv][c]);
            if(nearlyZero(a[col][col])) return false;
            double inv=1.0/a[col][col];
            for(int c=0;c<6;c++) a[col][c]*=inv;
            for(int r=0;r<3;r++){
                if(r==col) continue;
                double f=a[r][col];
                for(int c=0;c<6;c++) a[r][c]-=f*a[col][c];
            }
        }
        for(int r=0;r<3;r++) for(int c=0;c<3;c++) out(r,c)=a[r][c+3];
        return true;
    }

    inline bool ApproxMat(const Mat3d& A, const Mat3d& B, double tol=kEps) {
        for(int i=0;i<3;i++) for(int j=0;j<3;j++)
            if(!approxEq(A(i,j),B(i,j),tol)) return false;
        return true;
    }

    // Matrice intrinseque camera
    inline Mat3d Intrinsics(double fx,double fy,double cx,double cy){
        Mat3d K=Mat3d::Identity();
        K(0,0)=fx; K(1,1)=fy; K(0,2)=cx; K(1,2)=cy; return K;
    }

    // Transformations 2D
    inline Mat3d Translate(const Vec2d& t){
        Mat3d T=Mat3d::Identity(); T(0,2)=t.x; T(1,2)=t.y; return T;
    }
    inline Mat3d Scale(const Vec2d& s){
        Mat3d S=Mat3d::Identity(); S(0,0)=s.x; S(1,1)=s.y; return S;
    }
    inline Mat3d TRS(const Vec2d& t,const Vec3d& ax,double ang,const Vec2d& s){
        return Translate(t)*Mat3d::RotateAxis(ax,ang)*Scale(s);
    }
    inline Mat3d TRS(const Vec2d& t,const Vec3d& r,const Vec2d& s){
        Mat3d rot=Mat3d::RotateAxis({0,0,1},r.z)
                 *Mat3d::RotateAxis({0,1,0},r.y)
                 *Mat3d::RotateAxis({1,0,0},r.x);
        return Translate(t)*rot*Scale(s);
    }

    // Decomposition TRS 2D
    inline void DecomposeTRS(const Mat3d& M,Vec2d& outT,double& outR,Vec2d& outS){
        outT={M(0,2),M(1,2)};
        Vec2d c0{M(0,0),M(1,0)}, c1{M(0,1),M(1,1)};
        double sx=c0.Norm(), sy=c1.Norm();
        outS={sx,sy};
        outR=std::atan2(M(1,0)/sx, M(0,0)/sx);
    }

} // namespace NkMath

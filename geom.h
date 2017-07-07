#ifndef GEOM_H
#define GEOM_H

#include <math.h>
#include <vector>

#define EPSILON 0.00000001

#define dot(u,v)   ((u).x * (v).x + (u).y * (v).y + (u).z * (v).z) // dot product (3D) which allows vector operations in arguments
#define norm(v)    sqrt(dot((v),(v)))  // norm = length of  vector
#define d(u,v)     norm((u)-(v))        // distance = norm of difference
#define abs(x)     ((x) >= 0 ? (x) : -(x))   //  absolute value
#define cross(u,v) (Vector((u).y*(v).z - (u).z*(v).y, (u).z*(v).x - (u).x*(v).z, (u).x*(v).y - (u).y*(v).x)) //  cross product
#define sign(x) ((x) > 0 ? 1 : ((x) < 0 ? -1 : 0))
#define perp(u,v)  ((u).x * (v).y - (u).y * (v).x)  // perp product  (2D)

class Vector
{
public:
    double x,y,z;
    
    Vector() : x(0),y(0),z(0) {};
    
    Vector(double _x, double  _y, double  _z) : x(_x),y(_y),z(_z) {};
    
    Vector(const Vector& v) : x(v.x),y(v.y),z(v.z) {};
    
//     virtual ~Vector() {}
    
    Vector operator+(const Vector& b) const
    {
        return Vector(x+b.x, y+b.y, z+b.z);
    }
    Vector operator-(const Vector& b) const
    {
        return Vector(x-b.x, y-b.y, z-b.z);
    }
    Vector operator*(const double& s) const
    {
        return Vector(x*s, y*s, z*s);
    }
    Vector operator/(const double& s) const
    {
        return Vector(x/s, y/s, z/s);
    }
    Vector& operator*=(const double& s)
    {
        x*=s; y*=s; z*=s;
        return *this;
    }
    Vector& operator/=(const double& s)
    {
        x/=s; y/=s; z/=s;
        return *this;
    }
    bool operator!=(const Vector& b) const
    {
        return x!=b.x || y!=b.y || z!=b.z;
    }
    bool operator==(const Vector& b) const
    {
        return x==b.x || y==b.y || z==b.z;
    }
};

inline Vector operator*(const double& s, const Vector& v) {
    return Vector(v.x*s, v.y*s, v.z*s);
}

class Lineseg
{
public:
    Lineseg() : P0(), P1() {};
    Lineseg(const Vector& _P0, const Vector& _P1) : P0(_P0), P1(_P1) {};
    
    Vector P0;
    Vector P1;
};

class Track
{
public:
    Track() : P0(), v() {};
    Track(const Vector& _P0, const Vector& _v) : P0(_P0), v(_v) {};
    
    Vector P0;
    Vector v; // velocity
};

class Ray
{
public:
    Ray() : P0(), dir() {};
    Ray(const Vector& _P0, const Vector& _dir, bool dirisnormalized=false)
    : P0(_P0), dir(_dir)
    {
        if (!dirisnormalized) {
            dir /= norm(dir);
        }
    }
    
    Vector P0;
    Vector dir; // direction (must be normalized)
};

class Triangle
{
public:
    Triangle() : P0(), P1(), P2() {};
    Triangle(const Vector& _P0, const Vector& _P1, const Vector& _P2)
    : P0(_P0), P1(_P1), P2(_P2) {};
    
    Vector P0;
    Vector P1;
    Vector P2;
};

enum Intersectiontype {
    IT_SEPARATE=0,
    IT_1IN2,
    IT_2IN1,
    IT_INTERSECT
};

enum Pointinpolytest {
    PIP_CROSSING=0,
    PIP_WINDING
};

double pointLinesegDistance2D(
        const Vector& P, const Lineseg& L,
        double& f);

int linesegLinesegIntersection2D(
        const Lineseg& S1, const Lineseg& S2,
        Vector& I0, Vector& I1);
bool inSegment2D(
        const Vector& P, const Lineseg& S);

double lineLineDistance3D(
        const Lineseg& L1, const Lineseg& L2,
        double& sc, double& tc);
double linesegLinesegDistance3D(
        const Lineseg& S1, const Lineseg& S2,
        double& sc, double& tc);
double pointLineDistance3D(
        const Vector& point, const Lineseg& lseg,
        double& t);
double pointLinesegDistance3D(
        const Vector& point, const Lineseg& lseg,
        double& t);
        
bool trianglesetTrianglesetIntersection3D(
        const std::vector<Triangle>& triset1,
        const std::vector<Triangle>& triset2);
bool triangleTriangleIntersection3D(
        const Triangle& tri1, const Triangle& tri2);
bool rayTriangleIntersection3D(
        const Ray& ray, const Triangle& tri,
        double& out, double& bary1, double& bary2, double& bary3);
double linesegTriangleDistance3D(
        const Lineseg& S, const Triangle& tri,
        double& t, double& bary1, double& bary2, double& bary3);
double pointTriangleDistance3D(
        const Vector& point, const Triangle& tri,
        double& bary1, double& bary2, double& bary3);

double pointPointTracktime3D(
        const Track& Tr1, const Track& Tr2);
double pointPointTrackdist3D(
        const Track& Tr1, const Track& Tr2);

void poinsetInPolygonset2D(
        std::vector<double>::const_iterator px,
        std::vector<double>::const_iterator pxend,
        std::vector<double>::const_iterator py,
        std::vector<double>::const_iterator pyend,
        std::vector<std::vector<double> >::const_iterator v,
        std::vector<std::vector<double> >::const_iterator vend,
        std::vector<bool>::iterator in,
        std::vector<bool>::iterator inend,
        Pointinpolytest piptest=PIP_CROSSING);
void poinsetInPolygonset2D(
        std::vector<double>::const_iterator px,
        std::vector<double>::const_iterator pxend,
        std::vector<double>::const_iterator py,
        std::vector<double>::const_iterator pyend,
        std::vector<std::vector<double> >::const_iterator vx,
        std::vector<std::vector<double> >::const_iterator vxend,
        std::vector<std::vector<double> >::const_iterator vy,
        std::vector<std::vector<double> >::const_iterator vyend,
        std::vector<bool>::iterator in,
        std::vector<bool>::iterator inend,
        Pointinpolytest piptest=PIP_CROSSING);

void pointsetInPolygon2D(std::vector<double>::const_iterator px,
        std::vector<double>::const_iterator pxend,
        std::vector<double>::const_iterator py,
        std::vector<double>::const_iterator pyend,
        std::vector<double>::const_iterator v,
        std::vector<double>::const_iterator vend,
        std::vector<bool>::iterator in,
        std::vector<bool>::iterator inend,
        Pointinpolytest piptest);
void pointsetInPolygon2D(
        std::vector<double>::const_iterator px,
        std::vector<double>::const_iterator pxend,
        std::vector<double>::const_iterator py,
        std::vector<double>::const_iterator pyend,
        std::vector<double>::const_iterator vx,
        std::vector<double>::const_iterator vxend,
        std::vector<double>::const_iterator vy,
        std::vector<double>::const_iterator vyend,
        std::vector<bool>::iterator in,
        std::vector<bool>::iterator inend,
        Pointinpolytest piptest);
        
bool pointInPolygon2D_crossing(
        double px, double py,
        std::vector<double>::const_iterator v,
        std::vector<double>::const_iterator vend);
bool pointInPolygon2D_crossing(
        double px, double py,
        std::vector<double>::const_iterator vx,
        std::vector<double>::const_iterator vxend,
        std::vector<double>::const_iterator vy,
        std::vector<double>::const_iterator vyend);

bool pointInPolygon2D_winding(
        double px, double py,
        std::vector<double>::const_iterator v,
        std::vector<double>::const_iterator vend);
bool pointInPolygon2D_winding(
        double px, double py,
        std::vector<double>::const_iterator vx,
        std::vector<double>::const_iterator vxend,
        std::vector<double>::const_iterator vy,
        std::vector<double>::const_iterator vyend);

void polygonBoundingbox2D(
        std::vector<double>::const_iterator v,
        std::vector<double>::const_iterator vend,
        double& minvx, double& minvy,
        double& maxvx, double& maxvy);
void polygonBoundingbox2D(
        std::vector<double>::const_iterator vx,
        std::vector<double>::const_iterator vxend,
        std::vector<double>::const_iterator vy,
        std::vector<double>::const_iterator vyend,
        double& minvx, double& minvy,
        double& maxvx, double& maxvy);

void polygonsetPolygonsetIntersection2D(
        std::vector<std::vector<double> >::const_iterator p1,
        std::vector<std::vector<double> >::const_iterator p1end,
        std::vector<std::vector<double> >::const_iterator p2,
        std::vector<std::vector<double> >::const_iterator p2end,
        std::vector<Intersectiontype>::iterator it,
        std::vector<Intersectiontype>::iterator itend);

inline double pointIsLeftofLine2D(
        const double& px, const double& py,
        const double& l1x, const double& l1y, const double& l2x, const double& l2y);

#endif // GEOM_H

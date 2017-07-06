// Copyright 2001 softSurfer, 2012 Dan Sunday
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.
 

// Assume that classes are already given for the objects:
//    Point and Vector with
//        coordinates {double x, y, z;}
//        operators for:
//            Point   = Point ± Vector
//            Vector =  Point - Point
//            Vector =  Vector ± Vector
//            Vector =  Scalar * Vector
//    Line and Lineseg with defining points {Point  P0, P1;}
//    Track with initial position and velocity vector
//            {Point P0;  Vector v;}
//===================================================================
 
#include <algorithm>
 
#include "triTriIntersectionDevillers.h"

#include "geom.h"

double pointLinesegDistance2D(const Vector& P, const Lineseg& S, double& f)
{
    bool out1,out2;
    if (S.P1.x==S.P0.x && S.P1.y==S.P0.y) { // degenerate segment (zero length)
        out1 = true;
        out2 = false;
    } else {
        out1 = ((S.P1.x-S.P0.x)*(P.x-S.P0.x)) + ((S.P1.y-S.P0.y)*(P.y-S.P0.y)) < 0; //(ps to p) projected to the segment (ps to pe)
        out2 = ((S.P1.x-S.P0.x)*(P.x-S.P1.x)) + ((S.P1.y-S.P0.y)*(P.y-S.P1.y)) > 0; //(pe to p) projected to the segment (ps to pe)  
    }
    
    double dsq;
    if (!out1 && !out2) { 
        double a = fabs(((S.P1.x-S.P0.x)*(S.P0.y-P.y)) - ((S.P1.y-S.P0.y)*(S.P0.x-P.x)));
        dsq =  (a*a)/(((S.P1.x-S.P0.x)*(S.P1.x-S.P0.x)) + ((S.P1.y-S.P0.y)*(S.P1.y-S.P0.y)));
        
        f = (((S.P1.x-S.P0.x)*(P.x-S.P0.x)) + ((S.P1.y-S.P0.y)*(P.y-S.P0.y))) /
            (((S.P1.x-S.P0.x)*(S.P1.x-S.P0.x)) + ((S.P1.y-S.P0.y)*(S.P1.y-S.P0.y)));
//        cp.x = (S.P0.x*(1.0-f)) + (S.P1.x*f);
//        cp.y = (S.P0.y*(1.0-f)) + (S.P1.y*f);
    } else if(out1) {
        dsq = ((P.x-S.P0.x)*(P.x-S.P0.x)) + ((P.y-S.P0.y)*(P.y-S.P0.y));
        f = 0.0;
//        cp = S.P0;
    } else {
        dsq = ((P.x-S.P1.x)*(P.x-S.P1.x)) + ((P.y-S.P1.y)*(P.y-S.P1.y));
        f = 1.0;
//        cp = S.P1;
    }
    
    return sqrt(dsq);
}

// intersect2D_2Segments(): find the 2D intersection of 2 finite segments
//    Input:  two finite segments S1 and S2
//    Output: *I0 = intersect point (when it exists)
//            *I1 =  endpoint of intersect segment [I0,I1] (when it exists)
//    Return: 0=disjoint (no intersect)
//            1=intersect  in unique point I0
//            2=overlap  in segment from I0 to I1
int linesegLinesegIntersection2D(const Lineseg& S1, const Lineseg& S2, Vector& I0, Vector& I1 )
{
    Vector    u = S1.P1 - S1.P0;
    Vector    v = S2.P1 - S2.P0;
    Vector    w = S1.P0 - S2.P0;
    double     D = perp(u,v);

    // test if  they are parallel (includes either being a point)
    if (fabs(D) < EPSILON) {           // S1 and S2 are parallel
        if (perp(u,w) != 0 || perp(v,w) != 0)  {
            return 0;                    // they are NOT collinear
        }
        // they are collinear or degenerate
        // check if they are degenerate  points
        double du = dot(u,u);
        double dv = dot(v,v);
        if (du==0 && dv==0) {            // both segments are points
            if (S1.P0 !=  S2.P0)         // they are distinct  points
                 return 0;
            I0 = S1.P0;                 // they are the same point
            return 1;
        }
        if (du==0) {                     // S1 is a single point
            if  (!inSegment2D(S1.P0, S2))  // but is not in S2
                 return 0;
            I0 = S1.P0;
            return 1;
        }
        if (dv==0) {                     // S2 a single point
            if  (!inSegment2D(S2.P0, S1))  // but is not in S1
                 return 0;
            I0 = S2.P0;
            return 1;
        }
        // they are collinear segments - get  overlap (or not)
        double t0, t1;                    // endpoints of S1 in eqn for S2
        Vector w2 = S1.P1 - S2.P0;
        if (v.x != 0) {
            t0 = w.x / v.x;
            t1 = w2.x / v.x;
        }
        else {
            t0 = w.y / v.y;
            t1 = w2.y / v.y;
        }
        if (t0 > t1) {                   // must have t0 smaller than t1
            double t=t0; t0=t1; t1=t;    // swap if not
        }
        if (t0 > 1 || t1 < 0) {
            return 0;      // NO overlap
        }
        t0 = t0<0? 0 : t0;               // clip to min 0
        t1 = t1>1? 1 : t1;               // clip to max 1
        if (t0 == t1) {                  // intersect is a point
            I0 = S2.P0 +  t0 * v;
            return 1;
        }

        // they overlap in a valid subsegment
        I0 = S2.P0 + t0 * v;
        I1 = S2.P0 + t1 * v;
        return 2;
    }

    // the segments are skew and may intersect in a point
    // get the intersect parameter for S1
    double     sI = perp(v,w) / D;
    if (sI < 0 || sI > 1)                // no intersect with S1
        return 0;

    // get the intersect parameter for S2
    double     tI = perp(u,w) / D;
    if (tI < 0 || tI > 1)                // no intersect with S2
        return 0;

    I0 = S1.P0 + sI * u;                // compute S1 intersect point
    return 1;
}
//===================================================================
 


// inSegment(): determine if a point is inside a segment
//    Input:  a point P, and a collinear segment S
//    Return: 1 = P is inside S
//            0 = P is  not inside S
bool inSegment2D(const Vector& P, const Lineseg& S)
{
    if (S.P0.x != S.P1.x) {    // S is not  vertical
        if (S.P0.x <= P.x && P.x <= S.P1.x)
            return true;
        if (S.P0.x >= P.x && P.x >= S.P1.x)
            return true;
    }
    else {    // S is vertical, so test y  coordinate
        if (S.P0.y <= P.y && P.y <= S.P1.y)
            return true;
        if (S.P0.y >= P.y && P.y >= S.P1.y)
            return true;
    }
    return false;
}
//===================================================================


// lineLineDistance3D(): get the 3D minimum distance between 2 lines
//    Input:  two 3D lines L1 and L2
//    Return: the shortest distance between L1 and L2
double lineLineDistance3D(const Lineseg& L1, const Lineseg& L2, double& sc, double& tc)
{
    Vector   u = L1.P1 - L1.P0;
    Vector   v = L2.P1 - L2.P0;
    Vector   w = L1.P0 - L2.P0;
    double    a = dot(u,u);         // always >= 0
    double    b = dot(u,v);
    double    c = dot(v,v);         // always >= 0
    double    d = dot(u,w);
    double    e = dot(v,w);
    double    D = a*c - b*b;        // always >= 0

    // compute the line parameters of the two closest points
    if (D < EPSILON) {          // the lines are almost parallel
        sc = 0.0;
        tc = (b>c ? d/b : e/c);    // use the largest denominator
    }
    else {
        sc = (b*e - c*d) / D;
        tc = (a*e - b*d) / D;
    }

    // get the difference of the two closest points
    Vector   dP = w + (sc * u) - (tc * v);  // =  L1(sc) - L2(tc)

    return norm(dP);   // return the closest distance
}

// segmentLinesegDistance3D(): get the 3D minimum distance between 2 segments
//    Input:  two 3D line segments S1 and S2
//    Return: the shortest distance between S1 and S2
double linesegLinesegDistance3D(const Lineseg& S1, const Lineseg& S2, double& sc, double& tc)
{
    Vector   u = S1.P1 - S1.P0;
    Vector   v = S2.P1 - S2.P0;
    Vector   w = S1.P0 - S2.P0;
    double    a = dot(u,u);         // always >= 0
    double    b = dot(u,v);
    double    c = dot(v,v);         // always >= 0
    double    d = dot(u,w);
    double    e = dot(v,w);
    double    D = a*c - b*b;        // always >= 0
    double    sN, sD = D;       // sc = sN / sD, default sD = D >= 0
    double    tN, tD = D;       // tc = tN / tD, default tD = D >= 0

    
    if (a < EPSILON) {
        // degenerate first line segment (is really a point)
        sc = 0;
        return pointLinesegDistance3D(S1.P0,S2,tc);
    } else if (c < EPSILON) {
        // degenerate second line segment (is really a point)
        tc = 0;
        return pointLinesegDistance3D(S2.P0,S1,sc);
    }

    // compute the line parameters of the two closest points
    if (D < EPSILON) { // the lines are almost parallel
        sN = 0.0;         // force using point P0 on segment S1
        sD = 1.0;         // to prevent possible division by 0.0 later
        tN = e;
        tD = c;
    }
    else {                 // get the closest points on the infinite lines
        sN = (b*e - c*d);
        tN = (a*e - b*d);
        if (sN < 0.0) {        // sc < 0 => the s=0 edge is visible
            sN = 0.0;
            tN = e;
            tD = c;
        }
        else if (sN > sD) {  // sc > 1  => the s=1 edge is visible
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }

    if (tN < 0.0) {            // tc < 0 => the t=0 edge is visible
        tN = 0.0;
        // recompute sc for this edge
        if (-d < 0.0)
            sN = 0.0;
        else if (-d > a)
            sN = sD;
        else {
            sN = -d;
            sD = a;
        }
    }
    else if (tN > tD) {      // tc > 1  => the t=1 edge is visible
        tN = tD;
        // recompute sc for this edge
        if ((-d + b) < 0.0)
            sN = 0;
        else if ((-d + b) > a)
            sN = sD;
        else {
            sN = (-d +  b);
            sD = a;
        }
    }
    // finally do the division to get sc and tc
    sc = (abs(sN) < EPSILON ? 0.0 : sN / sD);
    tc = (abs(tN) < EPSILON ? 0.0 : tN / tD);

    // get the difference of the two closest points
    Vector dP = w + (sc * u) - (tc * v);  // =  S1(sc) - S2(tc)

    return norm(dP);   // return the closest distance
}


// pointPointTracktime3D(): compute the time of CPA for two tracks
//    Input:  two tracks Tr1 and Tr2
//    Return: the time at which the two tracks are closest
double pointPointTracktime3D(const Track& Tr1, const Track& Tr2)
{
    Vector   dv = Tr1.v - Tr2.v;

    double    dv2 = dot(dv,dv);
    if (dv2 < EPSILON)      // the  tracks are almost parallel
        return 0.0;             // any time is ok.  Use time 0.

    Vector   w0 = Tr1.P0 - Tr2.P0;
    double    cpatime = -dot(w0,dv) / dv2;

    return cpatime;             // time of CPA
}


// pointPointTrackdist3D(): compute the distance at CPA for two tracks
//    Input:  two tracks Tr1 and Tr2
//    Return: the distance for which the two tracks are closest
double pointPointTrackdist3D(const Track& Tr1, const Track& Tr2)
{
    double    ctime = pointPointTracktime3D(Tr1, Tr2);
    Vector    P1 = Tr1.P0 + (ctime * Tr1.v);
    Vector    P2 = Tr2.P0 + (ctime * Tr2.v);

    return d(P1,P2);            // distance at CPA
}


//bool check_same_clock_dir(const Vector& pt1, const Vector& pt2, const Vector& pt3, const Vector& norm)
//{  
//    double testi, testj, testk;
//    double dotprod;
//    // normal of trinagle
//    testi = (((pt2.y - pt1.y)*(pt3.z - pt1.z)) - ((pt3.y - pt1.y)*(pt2.z - pt1.z)));
//    testj = (((pt2.z - pt1.z)*(pt3.x - pt1.x)) - ((pt3.z - pt1.z)*(pt2.x - pt1.x)));
//    testk = (((pt2.x - pt1.x)*(pt3.y - pt1.y)) - ((pt3.x - pt1.x)*(pt2.y - pt1.y)));
//
//    // Dot product with triangle normal
//    dotprod = testi*norm.x + testj*norm.y + testk.norm.z;//
//
//   //answer
//    if(dotprod < 0)
//        return false;
//    else
//        return true;
//}

// lineTriangleIntersection3D(): compute intersection between line and triangle (Möller–Trumbore)
//    Input:  line S and triangle tri
//    Return: true if intersction and distance from P0 of the line to the intersection point in out (negative if in other direction than P1)
// from: http://www.angelfire.com/fl/houseofbartlett/solutions/line2tri.html
//bool lineTriangleIntersection3D(const Lineseg& S, const Triangle& tri, double& out)
//{
//    Ray ray(S.P0,S.P1-S.P0);
//    
//    double V1x, V1y, V1z;
//    double V2x, V2y, V2z;
//    Vector norm;
//    double dotprod;
//    double t;
//    Vector pt_int;
//
//    // vector form triangle tri.P0 to tri.P1
//    V1x = tri.P1.x - tri.P0.x;
//    V1y = tri.P1.y - tri.P0.y;
//    V1z = tri.P1.z - tri.P0.z;
//
//    // vector form triangle tri.P1 to tri.P2
//    V2x = tri.P2.x - tri.P1.x;
//    V2y = tri.P2.y - tri.P1.y;
//    V2z = tri.P2.z - tri.P1.z;
//
//    // vector normal of triangle
//    norm.x = V1y*V2z-V1z*V2y;
//    norm.y = V1z*V2x-V1x*V2z;
//    norm.z = V1x*V2y-V1y*V2x;
//
//    // dot product of normal and line's vector if zero line is parallel to triangle
//    dotprod = norm.x*ray.dir.x + norm.y*ray.dir.y + norm.z*ray.dir.z;
//
//    if (dotprod < 0) {
//        //Find point of intersect to triangle plane.
//        //find t to intersect point
//        t = -(norm.x*(ray.P0.x-tri.P0.x)+norm.y*(ray.P0.y-tri.P0.y)+norm.z*(ray.P0.z-tri.P0.z))/
//             (norm.x*ray.dir.x+norm.y*ray.dir.y+norm.z*ray.dir.z);
//
//        // if ds is neg line started past triangle so can't hit triangle.
//        if (t < 0)
//            return false
//
//        pt_int.x = ray.P0.x + ray.dir.x*t;
//        pt_int.y = ray.P0.y + ray.dir.y*t;
//        pt_int.z = ray.P0.z + ray.dir.z*t;
//
//        if (check_same_clock_dir(tri.P0, tri.P1, pt_int, norm)) {
//            if (check_same_clock_dir(tri.P1, tri.P2, pt_int, norm)) {
//                if (check_same_clock_dir(tri.P2, tri.P0, pt_int, norm)) {
//                    // answer in pt_int is insde triangle
//                    out = t;
//                    return true;
//                }
//            }
//        }
//   }
//   return false;
//}

bool trianglesetTrianglesetIntersection3D(const std::vector<Triangle>& triset1, const std::vector<Triangle>& triset2)
{
    int c1 = 0;
    std::vector<Triangle>::const_iterator i1,i2;
    for (i1 = triset1.begin(); i1!=triset1.end(); i1++) {
        int c2 = 0;
        for (i2 = triset2.begin(); i2!=triset2.end(); i2++) {
            if (triangleTriangleIntersection3D(*i1,*i2)) {
                return true;
            }
            c2++;
        }
        c1++;
    }
    return false;
}

bool triangleTriangleIntersection3D(const Triangle& tri1, const Triangle& tri2)
{
    double t1p1[3] = {tri1.P0.x,tri1.P0.y,tri1.P0.z};
    double t1p2[3] = {tri1.P1.x,tri1.P1.y,tri1.P1.z};
    double t1p3[3] = {tri1.P2.x,tri1.P2.y,tri1.P2.z};
    double t2p1[3] = {tri2.P0.x,tri2.P0.y,tri2.P0.z};
    double t2p2[3] = {tri2.P1.x,tri2.P1.y,tri2.P1.z};
    double t2p3[3] = {tri2.P2.x,tri2.P2.y,tri2.P2.z};
    
    return tri_tri_overlap_test_3d(t1p1,t1p2,t1p3,t2p1,t2p2,t2p3) == 1;
}

// rayTriangleIntersection3D(): compute intersection between ray and triangle (Möller–Trumbore)
//    Input:  ray ray and triangle tri
//    Return: true if intersction and multiples of ray direction to reach intersection point in out and barycentric coordintes of intersection point
bool rayTriangleIntersection3D(const Ray& ray, const Triangle& tri, double& out, double& bary1, double& bary2, double& bary3)
{
  Vector e1, e2;  //Edge1, Edge2
  Vector P, Q, T;
  double det, inv_det, u, v;
  double t;
 
  //Find vectors for two edges sharing tri.P0
  e1 = tri.P1 - tri.P0;
  e2 = tri.P2 - tri.P0;
  //Begin calculating determinant - also used to calculate u parameter
  P = cross(ray.dir, e2);
  //if determinant is near zero, ray lies in plane of triangle
  det = dot(e1, P);
  //NOT CULLING
  if(det > -EPSILON && det < EPSILON) return false;
  inv_det = 1.f / det;
 
  //calculate distance from tri.P0 to ray origin
  T = ray.P0 - tri.P0;
 
  //Calculate u parameter and test bound
  u = dot(T, P) * inv_det;
  //The intersection lies outside of the triangle
  if(u < 0.f || u > 1.f) return false;
 
  //Prepare to test v parameter
  Q = cross(T, e1);
 
  //Calculate V parameter and test bound
  v = dot(ray.dir, Q) * inv_det;
  //The intersection lies outside of the triangle
  if(v < 0.f || u + v  > 1.f) return false;
 
  t = dot(e2, Q) * inv_det;
 
  if(t > EPSILON) { //ray intersection
    out = t;
    bary1 = 1.0-(u+v);
    bary2 = u;
    bary3 = v;
    return true;
  }
 
  // No hit, no win
  return false;
}

//void barycentricCoords(const Vector& p, const Traingle& tri, double& bary1, double& bary2, double& bary3)
//{
//    Vector p1 = tri.P0-p;
//    Vector p2 = tri.P1-p;
//    Vector p3 = tri.P2-p;
//    // calculate the areas (parameters order is essential in this case):
//    Vector va = cross(tri.P0-tri.P1, tri.P0-tri.P2); // main triangle cross product
//    Vector va1 = cross(p2, p3); // tri.P0's triangle cross product
//    Vector va2 = cross(p3, p1); // tri.P1's triangle cross product
//    Vector va3 = cross(p1, p2); // tri.P2's triangle cross product
//    double a = norm(va); // main triangle area
//    // calculate barycentric coordinates with sign:
//    bary1 = norm(va1)/a * sign(dot(va, va1));
//    bary2 = norm(va2)/a * sign(dot(va, va2));
//    bary3 = norm(va3)/a * sign(dot(va, va3));
//}

// linesegTriangleDistance3D(): compute minimum distance between line segment and triangle
//    Input:  line segment S and triangle tri
//    Return: the shortest distance between S and tri and multiples of the linseg in [0,1] to the closest point in t
double linesegTriangleDistance3D(const Lineseg& S, const Triangle& tri, double& t, double& bary1, double& bary2, double& bary3)
{
    double dist = -1;
    t = -1;
    
    Vector e1(tri.P1 - tri.P0);
    Vector e2(tri.P2 - tri.P0);
    Vector trinormal(cross(e1,e2));
    trinormal /= norm(trinormal);
    
    double edist,t1,t2;
    double ebary1,ebary2,ebary3;
    
    // check if lineseg and triangle intersect
    if (rayTriangleIntersection3D(Ray(S.P0,S.P1-S.P0), tri, t1, ebary1,ebary2,ebary3)) {
        double lineseglen = d(S.P0,S.P1);
        if (t1 <= lineseglen) {
            t = t1 / lineseglen; // normalize t to the lineseg length
            // Vector closestpoint(S.P0 + ((S.P1-S.P0)*t));
            // convert to barycentric coordinates
            bary1 = ebary1;
            bary2 = ebary2;
            bary3 = ebary3;
            return 0;
        }
    }

//    Vector3 areavec = cross(tri.P1-tri.P0,tri.P2-tri.P0);
//    double area = 0.5*sqrt(dot(areavec,areavec));
//    if (area > 0.000000001) {
        // check if point 1 is inside the triangle prism and get distance to triangle
        edist = pointTriangleDistance3D(S.P0,tri,ebary1,ebary2,ebary3);
        //if (edist < dist) {
            dist = edist;
            t = 0;
            bary1 = ebary1;
            bary2 = ebary2;
            bary3 = ebary3;
        //}
    
        // check if point 2 is inside the triangle prism and get distance to triangle
        edist = pointTriangleDistance3D(S.P1,tri,ebary1,ebary2,ebary3);
        if (edist < dist) {
            dist = edist;
            t = 1;
            bary1 = ebary1;
            bary2 = ebary2;
            bary3 = ebary3;
        }
//    } else {
//        edist = 
//    }

    // distance to edge1
    edist = linesegLinesegDistance3D(S,Lineseg(tri.P0,tri.P1),t1,t2);
    if (edist < dist) {
        dist = edist;
        t = t1;
        bary1 = 1.0-t2;
        bary2 = t2;
        bary3 = 0;
    }
    
    // distance to edge2
    edist = linesegLinesegDistance3D(S,Lineseg(tri.P1,tri.P2),t1,t2);
    if (edist < dist) {
        dist = edist;
        t = t1;
        bary1 = 0;
        bary2 = 1.0-t2;
        bary3 = t2;
    }
    
    // distance to edge3
    edist = linesegLinesegDistance3D(S,Lineseg(tri.P2,tri.P0),t1,t2);
    if (edist < dist) {
        dist = edist;
        t = t1;
        bary1 = t2;
        bary2 = 0;
        bary3 = 1.0-t2;
    }
    
    return dist;
}

double pointLineDistance3D(const Vector& point, const Lineseg& lseg, double& t)
{
    Vector segcenter = ((lseg.P0+lseg.P1)*0.5);
    Vector diff = point - segcenter;
    Vector segdir = lseg.P1-lseg.P0;
    double seglen = norm(segdir);
    
    if (seglen < EPSILON) {
        // lineseg is degenerate (really a point)
        t = 0;
        return norm(point - lseg.P0);
    }

    segdir /= seglen;
    t = dot(segdir,diff);
    
    Vector closestpoint = segcenter + t*segdir;
    
    return norm(closestpoint - point);
}

// Geometric Tools, LLC
// Copyright (c) 1998-2014
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.1 (2010/10/01)
double pointLinesegDistance3D(const Vector& point, const Lineseg& lseg, double& t)
{
    Vector segcenter = ((lseg.P0+lseg.P1)*0.5);
    Vector diff = point - segcenter;
    Vector segdir = lseg.P1-lseg.P0;
    double seglen = norm(segdir);
    
    if (seglen < EPSILON) {
        // lineseg is degenerate (really a point)
        t = 0;
        return norm(point - lseg.P0);
    }

    segdir /= seglen;
    t = dot(segdir,diff);
    
    Vector closestpoint;
    if (t > -seglen/2.0) {
        if (t < seglen/2.0) {
            closestpoint = segcenter + t*segdir;
        } else {
            closestpoint = lseg.P1;
            t = seglen/2.0;
        }
    } else {
        closestpoint = lseg.P0;
        t = -seglen/2.0;
    }
    
    t += seglen/2.0;
    t /= seglen;
    
    return norm(closestpoint - point);
}

// Geometric Tools, LLC
// Copyright (c) 1998-2014
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 5.0.1 (2010/10/01)
double pointTriangleDistance3D(const Vector& point, const Triangle& tri, double& bary1, double& bary2, double& bary3)
{
    Vector diff = tri.P0 - point;
    Vector edge0 = tri.P1 - tri.P0;
    Vector edge1 = tri.P2 - tri.P0;
    
    double a00 = dot(edge0,edge0);
    double a01 = dot(edge0,edge1);
    double a11 = dot(edge1,edge1);
    double b0 = dot(diff,edge0);
    double b1 = dot(diff,edge1);
    double c = dot(diff,diff);
    double det = abs(a00*a11 - a01*a01);
    double s = a01*b1 - a11*b0;
    double t = a01*b0 - a00*b1;
    double sqrDistance;

    if (abs(det) < EPSILON) {
        // triangle is degenerate (really a line segment)
        
        Vector e1(tri.P1-tri.P0);
        Vector e2(tri.P2-tri.P1);
        Vector e3(tri.P0-tri.P2);

        double e1sqlen = dot(e1,e1);
        double e2sqlen = dot(e2,e2);
        double e3sqlen = dot(e3,e3);

        // longest edge is the line segment
        double dist;
        if (e1sqlen > e2sqlen) {
            if (e1sqlen > e3sqlen) { // e1
                dist = pointLinesegDistance3D(point,Lineseg(tri.P0,tri.P1),t);
                bary3 = 0;
                bary1 = 1-t;
                bary2 = t;
            } else { // e2
                dist = pointLinesegDistance3D(point,Lineseg(tri.P1,tri.P2),t);
                bary1 = 0;
                bary2 = 1-t;
                bary3 = t;
            }
        } else {
            if (e2sqlen > e3sqlen) { // e2
                dist = pointLinesegDistance3D(point,Lineseg(tri.P1,tri.P2),t);
                bary1 = 0;
                bary2 = 1-t;
                bary3 = t;
            } else { // e3
                dist = pointLinesegDistance3D(point,Lineseg(tri.P2,tri.P0),t);
                bary2 = 0;
                bary3 = 1-t;
                bary1 = t;
            }
        }

        return dist;
    }

    if (s + t <= det) {
        if (s < 0) {
            if (t < 0) {  // region 4
                if (b0 < 0) {
                    t = 0;
                    if (-b0 >= a00) {
                        s = (double)1;
                        sqrDistance = a00 + ((double)2)*b0 + c;
                    } else {
                        s = -b0/a00;
                        sqrDistance = b0*s + c;
                    }
                } else {
                    s = 0;
                    if (b1 >= 0) {
                        t = 0;
                        sqrDistance = c;
                    } else if (-b1 >= a11) {
                        t = (double)1;
                        sqrDistance = a11 + ((double)2)*b1 + c;
                    } else {
                        t = -b1/a11;
                        sqrDistance = b1*t + c;
                    }
                }
            } else { // region 3
                s = 0;
                if (b1 >= 0) {
                    t = 0;
                    sqrDistance = c;
                } else if (-b1 >= a11) {
                    t = (double)1;
                    sqrDistance = a11 + ((double)2)*b1 + c;
                } else {
                    t = -b1/a11;
                    sqrDistance = b1*t + c;
                }
            }
        } else if (t < 0) { // region 5
            t = 0;
            if (b0 >= 0) {
                s = 0;
                sqrDistance = c;
            } else if (-b0 >= a00) {
                s = (double)1;
                sqrDistance = a00 + ((double)2)*b0 + c;
            } else {
                s = -b0/a00;
                sqrDistance = b0*s + c;
            }
        } else  { // region 0
            // minimum at interior point
            double invDet = ((double)1)/det;
            s *= invDet;
            t *= invDet;
            sqrDistance = s*(a00*s + a01*t + ((double)2)*b0) +
                t*(a01*s + a11*t + ((double)2)*b1) + c;
        }
    } else {
        double tmp0, tmp1, numer, denom;

        if (s < 0) { // region 2
            tmp0 = a01 + b0;
            tmp1 = a11 + b1;
            if (tmp1 > tmp0) {
                numer = tmp1 - tmp0;
                denom = a00 - ((double)2)*a01 + a11;
                if (numer >= denom) {
                    s = (double)1;
                    t = 0;
                    sqrDistance = a00 + ((double)2)*b0 + c;
                } else {
                    s = numer/denom;
                    t = (double)1 - s;
                    sqrDistance = s*(a00*s + a01*t + ((double)2)*b0) +
                        t*(a01*s + a11*t + ((double)2)*b1) + c;
                }
            } else {
                s = 0;
                if (tmp1 <= 0) {
                    t = (double)1;
                    sqrDistance = a11 + ((double)2)*b1 + c;
                } else if (b1 >= 0) {
                    t = 0;
                    sqrDistance = c;
                } else {
                    t = -b1/a11;
                    sqrDistance = b1*t + c;
                }
            }
        } else if (t < 0) { // region 6
            tmp0 = a01 + b1;
            tmp1 = a00 + b0;
            if (tmp1 > tmp0)  {
                numer = tmp1 - tmp0;
                denom = a00 - ((double)2)*a01 + a11;
                if (numer >= denom) {
                    t = (double)1;
                    s = 0;
                    sqrDistance = a11 + ((double)2)*b1 + c;
                } else {
                    t = numer/denom;
                    s = (double)1 - t;
                    sqrDistance = s*(a00*s + a01*t + ((double)2)*b0) +
                        t*(a01*s + a11*t + ((double)2)*b1) + c;
                }
            } else {
                t = 0;
                if (tmp1 <= 0) {
                    s = (double)1;
                    sqrDistance = a00 + ((double)2)*b0 + c;
                } else if (b0 >= 0) {
                    s = 0;
                    sqrDistance = c;
                } else {
                    s = -b0/a00;
                    sqrDistance = b0*s + c;
                }
            }
        } else { // region 1
            numer = a11 + b1 - a01 - b0;
            if (numer <= 0) {
                s = 0;
                t = (double)1;
                sqrDistance = a11 + ((double)2)*b1 + c;
            } else {
                denom = a00 - ((double)2)*a01 + a11;
                if (numer >= denom) {
                    s = (double)1;
                    t = 0;
                    sqrDistance = a00 + ((double)2)*b0 + c;
                } else {
                    s = numer/denom;
                    t = (double)1 - s;
                    sqrDistance = s*(a00*s + a01*t + ((double)2)*b0) +
                        t*(a01*s + a11*t + ((double)2)*b1) + c;
                }
            }
        }
    }

    // Account for numerical round-off error.
    if (sqrDistance < 0) {
        sqrDistance = 0;
    }

    // to barycentric coordinates
    bary1 = (double)1 - s - t;
    bary2 = s;
    bary3 = t;

    return sqrt(sqrDistance);
}

// void poinsetInPolygonset2D(
//         const std::vector<double>& px,
//         const std::vector<double>& py,
//         const std::vector<std::vector<double> >& v,
//         std::vector<bool>& in,
//         Pointinpolytest piptest=PIP_CROSSING)
void poinsetInPolygonset2D(
        std::vector<double>::const_iterator px,
        std::vector<double>::const_iterator pxend,
        std::vector<double>::const_iterator py,
        std::vector<double>::const_iterator pyend,
        std::vector<std::vector<double> >::const_iterator v,
        std::vector<std::vector<double> >::const_iterator vend,
        std::vector<bool>::iterator in,
        std::vector<bool>::iterator inend,
        Pointinpolytest piptest)
{
    size_t npoints = pxend-px;
    for (; v!=vend && in<inend-(npoints-1); ++v,in+=npoints) {
        pointsetInPolygon2D(
                px,pxend,
                py,pyend,
                v->begin(),v->end(),
                in,in+npoints,
                piptest);
    }
}
        
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
        Pointinpolytest piptest)
{
    size_t npoints = pxend-px;
    for (; vx!=vxend && vy!=vyend && in<inend-(npoints-1); ++vx,++vy,in+=npoints) {
        pointsetInPolygon2D(
                px,pxend,
                py,pyend,
                vx->begin(),vx->end(),
                vy->begin(),vy->end(),
                in,in+npoints,
                piptest);
    }
}

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
        Pointinpolytest piptest)
{
    // compute polygon boundingbox
    double minvx, minvy, maxvx, maxvy;
    polygonBoundingbox2D(vx, vxend, vy, vyend, minvx, minvy, maxvx, maxvy);
    
    if (piptest==PIP_CROSSING) {
        for(; px!=pxend && py!=pyend && in!=inend; ++px, ++py, ++in) {

            // check if point is inside polygon boundingbox
            if (*px >= minvx && *px <= maxvx && *py >= minvy && *py <= maxvy) {
                *in = pointInPolygon2D_crossing(*px,*py,vx,vxend,vy,vyend);
            } else {
                *in = false;
            }
        }
    } else if (piptest==PIP_WINDING) {
        for(; px!=pxend && py!=pyend && in!=inend; ++px, ++py, ++in) {

            // check if point is inside polygon boundingbox
            if (*px >= minvx && *px <= maxvx && *py >= minvy && *py <= maxvy) {
                *in = pointInPolygon2D_winding(*px,*py,vx,vxend,vy,vyend);
            } else {
                *in = false;
            }
        }
    } else {
        // inrecognized inside/outside test, set to not inside
        for(; in!=inend; ++in) {
            *in = false;
        }
    }
}

void pointsetInPolygon2D(
        std::vector<double>::const_iterator px,
        std::vector<double>::const_iterator pxend,
        std::vector<double>::const_iterator py,
        std::vector<double>::const_iterator pyend,
        std::vector<double>::const_iterator v,
        std::vector<double>::const_iterator vend,
        std::vector<bool>::iterator in,
        std::vector<bool>::iterator inend,
        Pointinpolytest piptest)
{
    // compute polygon boundingbox
    double minvx, minvy, maxvx, maxvy;
    polygonBoundingbox2D(v, vend, minvx, minvy, maxvx, maxvy);
    
    if (piptest==PIP_CROSSING) {
        for(; px!=pxend && py!=pyend && in!=inend; ++px, ++py, ++in) {
            // check if point is inside polygon boundingbox
            if (*px >= minvx && *px <= maxvx && *py >= minvy && *py <= maxvy) {
                *in = pointInPolygon2D_crossing(*px,*py,v,vend);
            } else {
                *in = false;
            }
        }
    } else if (piptest==PIP_WINDING) {
        for(; px!=pxend && py!=pyend && in!=inend; ++px, ++py, ++in) {
            // check if point is inside polygon boundingbox
            if (*px >= minvx && *px <= maxvx && *py >= minvy && *py <= maxvy) {
                *in = pointInPolygon2D_winding(*px,*py,v,vend);
            } else {
                *in = false;
            }
        }
    } else {
        // inrecognized inside/outside test, set to not inside
        for(; in!=inend; ++in) {
            *in = false;
        }
    }
}

bool pointInPolygon2D_crossing(
        double px, double py,
        std::vector<double>::const_iterator vx,
        std::vector<double>::const_iterator vxend,
        std::vector<double>::const_iterator vy,
        std::vector<double>::const_iterator vyend)
{
    bool inside = false;
    
    if (vx>=vxend || vy>=vyend) {
        // polygon has no vertices
        return inside;
    }
    
    // check if point is inside polygon
    std::vector<double>::const_iterator lastvx = vxend-1;
    std::vector<double>::const_iterator lastvy = vyend-1;
    for (; vx!=vxend && vy!=vyend; ++vx, ++vy) {

        if (((*vy > py) != (*lastvy > py)) &&
            (px < (*lastvx - *vx) * (py - *vy) / (*lastvy - *vy) + *vx)) {
            inside = !inside;
        }

        lastvx = vx;
        lastvy = vy;
    }
    return inside;
}

bool pointInPolygon2D_crossing(
        double px, double py,
        std::vector<double>::const_iterator v,
        std::vector<double>::const_iterator vend)
{
    bool inside = false;
    
    if (v>=vend-1) {
        // polygon has no vertices
        return inside;
    }

    // check if point is inside polygon
    std::vector<double>::const_iterator lastv = vend-2;
    for (; v<vend-1; v+=2) {
        
        if (((*(v+1) > py) != (*(lastv+1) > py)) &&
            (px < (*lastv - *v) * (py - *(v+1)) / (*(lastv+1) - *(v+1)) + *v) ) {
            inside = !inside;
        }

        lastv = v;
    }
    
    return inside;
}

// vx and vy must not be empty! (otherwise outputs are not set)
void polygonBoundingbox2D(
        std::vector<double>::const_iterator vx,
        std::vector<double>::const_iterator vxend,
        std::vector<double>::const_iterator vy,
        std::vector<double>::const_iterator vyend,
        double& minvx, double& minvy,
        double& maxvx, double& maxvy)
{
    
    if (vx>=vxend || vy>=vyend) {
        // polygon has no vertices
        return;
    }
    
    minvx = *std::min_element(vx,vxend);
    minvy = *std::min_element(vy,vyend);
    maxvx = *std::max_element(vx,vxend);
    maxvy = *std::max_element(vy,vyend);
}

// v must have at least two elements! (otherwise outputs are not set)
void polygonBoundingbox2D(
        std::vector<double>::const_iterator v,
        std::vector<double>::const_iterator vend,
        double& minvx, double& minvy,
        double& maxvx, double& maxvy)
{
    if (v>=vend-1) {
        // polygon has no vertices
        return;
    }
    
    minvx = maxvx = v[0];
    minvy = maxvy = v[1];
    v+=2;
    for (; v<vend-1; v+=2) {
        if (*v < minvx) {
            minvx = *v;
        } else if (*v > maxvx) {
            maxvx = *v;
        }
        if (*(v+1) < minvy) {
            minvy = *(v+1);
        } else if (*(v+1) > maxvy) {
            maxvy = *(v+1);
        }
    }
}

//         const std::vector<std::vector<double> >& p1,
//         const std::vector<std::vector<double> >& p2,

void polygonsetPolygonsetIntersection2D(
        std::vector<std::vector<double> >::const_iterator p1,
        std::vector<std::vector<double> >::const_iterator p1end,
        std::vector<std::vector<double> >::const_iterator p2,
        std::vector<std::vector<double> >::const_iterator p2end,
        std::vector<Intersectiontype>::iterator it,
        std::vector<Intersectiontype>::iterator itend)
{
    Lineseg lseg1;
    Lineseg lseg2;
    Vector ip0;
    Vector ip1;
    
    std::vector<std::vector<double> >::const_iterator p1begin = p1;
    
    std::vector<double>::const_iterator v1, v2, lastv1, lastv2;
	for (; p2!=p2end && it!=itend; ++p2) { // over all polygons in second set
        for (p1=p1begin; p1!=p1end && it!=itend; ++p1, ++it) { // over all polygons in first set
            
            // handle degenerate cases first
            if (p1->size() < 4 || p2->size() < 4) {
                if (p1->size() < 2 || p2->size() < 2) {
                    // one of the polygons is empty
                    *it = IT_SEPARATE;
                } else if (p1->size() < 4 && p2->size() < 4) {
                    // both are points
                    if ((*p1)[0] == (*p2)[0] && (*p1)[1] == (*p2)[1]) {
                        // the points are the same
                        *it = IT_INTERSECT;
                    } else {
                        // the points are not the same
                        *it = IT_SEPARATE;
                    }
                } else {
                    if (p1->size() < 4) {
                        // first is a point, second one a polygon
                        if (pointInPolygon2D_crossing((*p1)[0],(*p1)[1],p2->begin(),p2->end())) {
                            *it = IT_1IN2;
                        } else {
                            *it = IT_SEPARATE;
                        }
                    } else {
                        // second is a point, first one a polygon
                        if (pointInPolygon2D_crossing((*p2)[0],(*p2)[1],p1->begin(),p1->end())) {
                            *it = IT_2IN1;
                        } else {
                            *it = IT_SEPARATE;
                        }
                    }
                }
            } else {
                // both are polygons
                *it = IT_SEPARATE;
                
                // compute bounding boxes
                double minv1x, minv1y, maxv1x, maxv1y;
                double minv2x, minv2y, maxv2x, maxv2y;
                polygonBoundingbox2D(p1->begin(), p1->end(), minv1x, minv1y, maxv1x, maxv1y);
                polygonBoundingbox2D(p2->begin(), p2->end(), minv2x, minv2y, maxv2x, maxv2y);
                
                // compute intersections if bounding boxes overlap
                if (abs((minv1x+maxv1x)-(minv2x+maxv2x)) <= (maxv1x-minv1x)+(maxv2x-minv2x) &&
                    abs((minv1y+maxv1y)-(minv2y+maxv2y)) <= (maxv1y-minv1y)+(maxv2y-minv2y)) {
                    
                    lastv1 = p1->begin()+(p1->size()-2);
                    for (v1=p1->begin(); v1<p1->end()-1 && *it != IT_INTERSECT; v1+=2) { // over all vertices in polygon 1
                        
                        lseg1.P0.x = *lastv1;
                        lseg1.P0.y = *(lastv1+1);
                        lseg1.P0.z = 0.0;
                        
                        lseg1.P1.x = *v1;
                        lseg1.P1.y = *(v1+1);
                        lseg1.P1.z = 0.0;
                        
                        lastv2 = p2->begin()+(p2->size()-2);
						for (v2=p2->begin(); v2<p2->end()-1 && *it != IT_INTERSECT; v2+=2) { // over all vertices in polygon 2
                            
                            lseg2.P0.x = *lastv2;
                            lseg2.P0.y = *(lastv2+1);
                            lseg2.P0.z = 0.0;
                            
                            lseg2.P1.x = *v2;
                            lseg2.P1.y = *(v2+1);
                            lseg2.P1.z = 0.0;
                            
                            if (linesegLinesegIntersection2D(lseg1,lseg2,ip0,ip1) > 0) {
                                *it = IT_INTERSECT;
                            }

                            lastv2 = v2;
                        }
                        lastv1 = v1;
                    }
                    
                    if (*it != IT_INTERSECT) {
                        if (pointInPolygon2D_crossing((*p1)[0],(*p1)[1],p2->begin(),p2->end())) {
                            *it = IT_1IN2;
                        } else if (pointInPolygon2D_crossing((*p2)[0],(*p2)[1],p1->begin(),p1->end())) {
                            *it = IT_2IN1;
                        } else  {
                            *it = IT_SEPARATE;
                        }
                    }
                    
                }

            }
            
        }
    }
    
}

// Next 3 functions Copyright 2000 softSurfer, 2012 Dan Sunday
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.
 
// a Point is defined by its coordinates {int x, y;}
//===================================================================
 

// pointIsLeftofLine(): tests if a point is Left|On|Right of an infinite line.
//    Input:  three points P0, P1, and P2
//    Return: >0 for P2 left of the line through P0 and P1
//            =0 for P2  on the line
//            <0 for P2  right of the line
//    See: Algorithm 1 "Area of Triangles and Polygons"
inline double pointIsLeftofLine2D(
        const double& px, const double& py,
        const double& l1x, const double& l1y, const double& l2x, const double& l2y)
{
    return ((l2x - l1x) * (py - l1y)
          - (px -  l1x) * (l2y - l1y));
}

// pointInPolygon2D_winding(): winding number test for a point in a polygon
//      Input:   P = a point,
//               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
//      Return:  wn = the winding number (=0 only when P is outside)
bool pointInPolygon2D_winding(
        double px, double py,
        std::vector<double>::const_iterator v,
        std::vector<double>::const_iterator vend)
{
    int wn = 0; // the  winding number counter

    std::vector<double>::const_iterator lastv = vend-2;
    for (; v<vend-1; v+=2) {
        
        if (*(lastv+1) <= py) { // start vy <= py
            if (*(v+1) > py) { // an upward crossing
                if (pointIsLeftofLine2D(px,py,*(lastv),*(lastv+1),*(v),*(v+1)) > 0) {
                    // P left of  edge => have a valid up intersect
                    ++wn;
                }
            }
        } else { // start vy > py (no test needed)
            if (*(v+1) <= py) { // a downward crossing
                if (pointIsLeftofLine2D(px,py,*(lastv),*(lastv+1),*(v),*(v+1)) < 0) {
                    // P right of  edge => have a valid down intersect
                    --wn; 
                }
            }
        }
        
        lastv = v;
    }
    return wn!=0;
}

bool pointInPolygon2D_winding(
        double px, double py,
        std::vector<double>::const_iterator vx,
        std::vector<double>::const_iterator vxend,
        std::vector<double>::const_iterator vy,
        std::vector<double>::const_iterator vyend)
{
    int wn = 0; // the  winding number counter

    std::vector<double>::const_iterator lastvx = vxend-1;
    std::vector<double>::const_iterator lastvy = vyend-1;
    for (; vx!=vxend && vy!=vyend; ++vx, ++vy) {
        
        if (*lastvy <= py) { // start vy <= py
            if (*vy > py) { // an upward crossing
                if (pointIsLeftofLine2D(px,py,*lastvx,*lastvy,*vx,*vy) > 0) {
                    // P left of  edge => have a valid up intersect
                    ++wn;
                }
            }
        } else { // start vy > py (no test needed)
            if (*vy <= py) { // a downward crossing
                if (pointIsLeftofLine2D(px,py,*lastvx,*lastvy,*vx,*vy) < 0) {
                    // P right of  edge => have a valid down intersect
                    --wn; 
                }
            }
        }
        
        lastvx = vx;
        lastvy = vy;
    }
    return wn!=0;
}
//===================================================================


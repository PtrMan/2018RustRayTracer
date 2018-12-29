// License: MIT


// TODO< textures for BVH information >



float dot2( in vec3 v ) { return dot(v,v); }


vec3 abs3(vec3 v) {
    return vec3(abs(v.x),abs(v.y),abs(v.z));
}


// extend is like the radius of a sphere
bool testInsideAabb(vec3 p, vec3 aabbCenter, vec3 aabbExtend) {
    vec3 absDiff = abs3(p - aabbCenter);
    return
        absDiff.x <= aabbExtend.x && 
        absDiff.y <= aabbExtend.y && 
        absDiff.z <= aabbExtend.z;
}



///////////////////
// intersectors


// returns t and normal
vec4 iBox( in vec3 ro, in vec3 rd, in mat4 txx, in mat4 txi, in vec3 rad ) {
    // The MIT License
    // Copyright © 2014 Inigo Quilez
    // Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


    // ref http://iquilezles.org/www/articles/intersectors/intersectors.htm
    // src https://www.shadertoy.com/view/ld23DV


    // Ray-Box intersection, by convertig the ray to the local space of the box.
    //
    // If this was used to raytace many equally oriented boxes (say you are traversing
    // a BVH), then the transformations in line ?? and ?? could be skipped, as well as
    // the normal computation in line ??. One over the ray direction is usually accessible
    // as well in raytracers, so the division would go away in real world applications.

    // convert from ray to box space
    vec3 rdd = (txx*vec4(rd,0.0)).xyz;
    vec3 roo = (txx*vec4(ro,1.0)).xyz;

    // ray-box intersection in box space
    vec3 m = 1.0/rdd;
    vec3 n = m*roo;
    vec3 k = abs(m)*rad;
    
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
    
    if( tN > tF || tF < 0.0) return vec4(-1.0);

    vec3 nor = -sign(rdd)*step(t1.yzx,t1.xyz)*step(t1.zxy,t1.xyz);

    // convert to ray space
    
    nor = (txi * vec4(nor,0.0)).xyz;

    return vec4( tN, nor );
}


float sBox( in vec3 ro, in vec3 rd, in mat4 txx, in vec3 rad ) {    
    // The MIT License
    // Copyright © 2014 Inigo Quilez
    // Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    // ref http://iquilezles.org/www/articles/intersectors/intersectors.htm
    // src https://www.shadertoy.com/view/ld23DV

    vec3 rdd = (txx*vec4(rd,0.0)).xyz;
    vec3 roo = (txx*vec4(ro,1.0)).xyz;

    vec3 m = 1.0/rdd;
    vec3 n = m*roo;
    vec3 k = abs(m)*rad;
    
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;

    float tN = max( max( t1.x, t1.y ), t1.z );
    float tF = min( min( t2.x, t2.y ), t2.z );
    if( tN > tF || tF < 0.0) return -1.0;
    
    return tN;
}



vec2 iSphere2(in vec3 ro, in vec3 rd, in vec4 sph) {
    // ref https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
    // src https://www.shadertoy.com/view/4d2XWV
    
    // The MIT License
    // Copyright © 2014 Inigo Quilez
    // Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    
    vec3 oc = ro - sph.xyz;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - sph.w*sph.w;
    float h = b*b - c;
    if( h<0.0 ) return vec2(-1.0);
    return vec2(-b - sqrt(h), -b + sqrt(h));
}


float iSphere(in vec3 ro, in vec3 rd, in vec4 sph) {
    return iSphere2(ro, rd, sph).x;
}

// Intersection of a ray and a capped cone oriented in an arbitrary direction
vec4 iCappedCone( in vec3  ro, in vec3  rd, 
                  in vec3  pa, in vec3  pb, 
                  in float ra, in float rb )
{
    // Other intersectors: http://iquilezles.org/www/articles/intersectors/intersectors.htm
    
    // The MIT License
    // Copyright © 2016 Inigo Quilez
    // Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    
    // src https://www.shadertoy.com/view/llcfRf
    
    vec3  ba = pb - pa;
    vec3  oa = ro - pa;
    vec3  ob = ro - pb;
    
    float baba = dot(ba,ba);
    float rdba = dot(rd,ba);
    float oaba = dot(oa,ba);
    float obba = dot(ob,ba);
    
    //caps
    if( oaba<0.0 )
    {
        // example of delayed division
        if( dot2(oa*rdba-rd*oaba)<(ra*ra*rdba*rdba) )
        {
            return vec4(-oaba/rdba,-ba*inversesqrt(baba));
        }
    }
    else if( obba>0.0 )
    {
        // example of NOT delayed division
        float t =-obba/rdba;
        if( dot2(ob+rd*t)<(rb*rb) )
        {
            return vec4(t,ba*inversesqrt(baba));
        }
    }
    
    
    // body
    float rr = rb - ra;
    float hy = baba + rr*rr;
    vec3  oc = oa*rb - ob*ra;
    
    float ocba = dot(oc,ba);
    float ocrd = dot(oc,rd);
    float ococ = dot(oc,oc);
    
    float k2 = baba*baba      - hy*rdba*rdba; // the gap is rdrd which is 1.0
    float k1 = baba*baba*ocrd - hy*rdba*ocba;
    float k0 = baba*baba*ococ - hy*ocba*ocba;

    float h = k1*k1 - k2*k0;
    if( h<0.0 ) return vec4(-1.0);

    float t = (-k1-sign(rr)*sqrt(h))/(k2*rr);
    
    float y = oaba + t*rdba;
    if( y>0.0 && y<baba ) 
    {
        return vec4(t, normalize(baba*(baba*(oa+t*rd)-rr*ba*ra)
                                 -ba*hy*y));
    }
    
    return vec4(-1.0);
}

struct TriangleIntersection {
    vec3 n;
    
    float t;
    float u;
    float v;
};

// Triangle intersection. Returns { t, u, v }
TriangleIntersection iTriangle(in vec3 ro, in vec3 rd, in vec3 v0, in vec3 v1, in vec3 v2) {
    // src https://www.shadertoy.com/view/MlGcDz
    // ref http://iquilezles.org/www/articles/intersectors/intersectors.htm
    // The MIT License
    // Copyright © 2014 Inigo Quilez
    // Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    
    vec3 v1v0 = v1 - v0;
    vec3 v2v0 = v2 - v0;
    vec3 rov0 = ro - v0;

    // The four determinants above have lots of terms in common. Knowing the changing
    // the order of the columns/rows doesn't change the volume/determinant, and that
    // the volume is dot(cross(a,b,c)), we can precompute some common terms and reduce
    // it all to:
    vec3  n = cross( v1v0, v2v0 );
    vec3  q = cross( rov0, rd );
    float d = 1.0/dot( rd, n );
    float u = d*dot( -q, v2v0 );
    float v = d*dot(  q, v1v0 );
    float t = d*dot( -n, rov0 );

    if( u<0.0 || u>1.0 || v<0.0 || (u+v)>1.0 ) t = -1.0;
    //t = min(u, min(v, min(1.0-u-v, t))); // by user "thewhuiteambit"
        
    TriangleIntersection res;
    res.t = t;
    res.u = u;
    res.v = v;
    res.n = n;
    return res;
}




////////////////////
// matrix helpers

mat4 rotateY(float rad) {
    // MIT license
    // src https://github.com/yuichiroharai/glsl-y-rotate/blob/master/rotateY.glsl
    
    float c = cos(rad);
    float s = sin(rad);
    return mat4(
        c, 0.0, -s, 0.0, 
        0.0, 1.0, 0.0, 0.0, 
        s, 0.0, c, 0.0, 
        0.0, 0.0, 0.0, 1.0
    );
}

mat4 translate(vec3 v) {
    return mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        v.x, v.y, v.z, 1.0
    );
}

//////////////////////
// distance functions and manipulation

// prefix is sdFUNCTIONNAME where "sd" stands for "signed distance"
// could not choose "i" because we may use it for "intersect"

float sdSphere(in vec3 p, in vec3 pos, in float radius) {
    float dist = length(p-pos);
    return dist - radius;
}




// fast normal computation
// /param pnn evaluated fn at p+vec2(1.0,-1.0).xyy
// /param nnp evaluated fn at p+vec2(1.0,-1.0).yyx
// /param npn evaluated fn at p+vec2(1.0,-1.0).yxy
// /param ppp evaluated fn at p+vec2(1.0,-1.0).xxx
vec3 sdNormalFast(in float pnn, in float nnp, in float npn, in float ppp) {
    // ref https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
    
    const vec2 k = vec2(1,-1);
    
    return -normalize(
        k.xyy*pnn + 
        k.yyx*nnp + 
        k.yxy*npn + 
        k.xxx*ppp);
}

// compute sine wave pertubation along a (normalized) direction
float sdPertubeCos(in vec3 p, in vec3 dir, in float phase) {
    float dotRes = dot(p, dir);
    
    return cos(dotRes+phase);
}







// 1.0/sqrt(1+1+1)
# define NORM3 0.57735026919


// own code
/* commented because experimental
vec3 starPoint(vec3 p) {
    float a = dot(p, vec3(1.0,0.0,0.0));
    float b = dot(p, vec3(-NORM3, -NORM3, -NORM3));
    float c = dot(p, vec3(-NORM3, 0.0, NORM3));
    float d = dot(p, vec3(-NORM3, 1.0, -NORM3));
    
    // limit ranges so that only positive length remain and are valid
    // at least one range has to be positive!
    float a2 = max(a, 0.0);
    float b2 = max(b, 0.0);
    float c2 = max(c, 0.0);
    float d2 = max(d, 0.0);
    
    int idx = 0;
    float maxValue = a2;
    vec3 dir = vec3(1.0,0.0,0.0);
    if (b2 > maxValue) {
        idx = 1;
        maxValue = b2;
        dir = vec3(-NORM3, -NORM3, -NORM3);
    }
    if (c2 > maxValue) {
        idx = 2;
        maxValue = c2;
        dir = vec3(-NORM3, 0.0, NORM3);
    }
    if (d2 > maxValue) {
        idx = 3;
        maxValue = d2;
        dir = vec3(-NORM3, 1.0, -NORM3);
    }
    
    float scale = maxValue;
    
    // limit to top cap of star-cylinders
    scale = min(scale, 1.0);
    
    return dir*scale;
}
*/









// triangle mesh intersection

// (global) merged mesh vertices of the scene
const vec4 meshVertices[4] = vec4[4](
    vec4(0.0, 0.0, -NORM3, 1.0),
    
    vec4(NORM3, 0.0, NORM3, 1.0),
    vec4(-NORM3, NORM3, NORM3, 1.0),
    vec4(-NORM3, -NORM3, NORM3, 1.0)
);

// (global) indirection of triangles to vertex indices
const int meshTriangleVertexIndices[3*4] = int[3*4](
    1, 2, 3,// bottom
    
    // sides
    1, 0, 2,
    2, 0, 3,
    3, 0, 1
);


struct MeshIntersection {
    TriangleIntersection triangle;
    int triangleIdx;
    float intersectionT;
};

// tests a ray against a mesh made up of triangles
// indices to the vertices are stored in meshTriangleVertexIndices
// triangles are stored in meshVertices, referenced by the vertex indices
    
// TODO< transform result normal by inverse matrix! >
MeshIntersection rayVsMesh(vec3 rayOrigin, vec3 rayDirection, int meshTriangleFirstIdx, int meshTrianglesCount, mat4 mat) {
    MeshIntersection result;
    result.triangleIdx = -1; // no triangle hit
    result.intersectionT = -1.0;
    
    for(
        int iMeshTriangleIdx = 0;
        iMeshTriangleIdx < meshTrianglesCount;
        iMeshTriangleIdx++
    ) {
        // fetch vertex indices
        int vertexIdx0 = meshTriangleVertexIndices[(meshTriangleFirstIdx+iMeshTriangleIdx)*3 + 0];
        int vertexIdx1 = meshTriangleVertexIndices[(meshTriangleFirstIdx+iMeshTriangleIdx)*3 + 1];
        int vertexIdx2 = meshTriangleVertexIndices[(meshTriangleFirstIdx+iMeshTriangleIdx)*3 + 2];
        
        // fetch vertex positions
        vec4 vertexPosition0 = mat * meshVertices[vertexIdx0];
        vec4 vertexPosition1 = mat * meshVertices[vertexIdx1];
        vec4 vertexPosition2 = mat * meshVertices[vertexIdx2];
        
        // check ray
        TriangleIntersection intersectionResult = iTriangle(
            rayOrigin, rayDirection,
            vec3(vertexPosition0.x, vertexPosition0.y, vertexPosition0.z),
            vec3(vertexPosition1.x, vertexPosition1.y, vertexPosition1.z),
            vec3(vertexPosition2.x, vertexPosition2.y, vertexPosition2.z)
        );
        float thisIntersection = intersectionResult.t;
        
        bool isCloserIntersection = false;
        
        if (thisIntersection >= 0.0 ) {
            // if it is the first intersection
            if (result.intersectionT < 0.0) {
                isCloserIntersection = true;
                
            }
            else {
                if (thisIntersection < result.intersectionT ) {
                    isCloserIntersection = true;
                }
                
            }
            
        }
        
        if (isCloserIntersection) {
            result.intersectionT = thisIntersection;
            result.triangle = intersectionResult;
            result.triangleIdx = iMeshTriangleIdx;

        }
    }
    
    result.triangle.n = normalize(result.triangle.n);
    return result;
}









///////////////////////////////
// BVH







// we store the BVH in a SoA because it has to be accessible by the host
//  * store the index of the left children node, is -1 if invalid 
//uniform int bvhNodeChildrenLeft[];

//  * store the index of the right children node, is -1 if invalid 
//uniform int bvhNodeChildrenRight[];

//  * is the specific bvh node a leaf
//uniform int bvhIsLeaf[];

//uniform vec4 bvhAabbCenter[];

//uniform vec4 bvhAabbExtend[];

// * indices to bvh leaf nodes in global array
//uniform int bvhLeafNodeIndices[];

uniform int bvhRootNodeIdx; // root node is at index 0

struct BvhNodeStruct {
    int nodeChildrenLeft;
    int nodeChildrenRight;
    int isLeaf; // bool
    int leafNodeIdx;

    vec4 aabbCenter;
    vec4 aabbExtend;
};

layout (std430, binding=0) buffer bvhNode {
    BvhNodeStruct bvhNodes[];
};






// bvh leaf nodes
// we store leaf nodes of the bvh in a global SoA

// type of node
// 0 : raytraced sphere
// 1 : polygon
// TODO< other types

uniform int bvhLeafNodeType[];

// position/(radius or attribute) encoding or first vertex
uniform vec4 bvhLeafNodeVertex0[];

// 2nd vertex
uniform vec4 bvhLeafNodeVertex1[];

// 2rd vertex
uniform vec4 bvhLeafNodeVertex2[];





// structure used to transribe the SoA to for simpler and more portable code
struct BvhNodeInfo2 {
    int childrenLeft;
    int childrenRight;
    
    vec3 aabbCenter;
    vec3 aabbExtend;
    
    bool isLeaf;
    
    int leafNodeIdx; // index to leaf node element in global array
};



// utility helper to read a BVH node from SoA to a structure
BvhNodeInfo2 retBvhNodeAt(int idx) {
    BvhNodeInfo2 res;
    res.childrenLeft = bvhNodes[idx].nodeChildrenLeft;
    res.childrenRight = bvhNodes[idx].nodeChildrenRight;
    res.isLeaf = bvhNodes[idx].isLeaf != 0;
    res.aabbCenter = bvhNodes[idx].aabbCenter.xyz;
    res.aabbExtend = bvhNodes[idx].aabbExtend.xyz;
    res.leafNodeIdx = bvhNodes[idx].leafNodeIdx;
    return res;
}


/*





// structure for iterative BVH traversal
struct BvhStackElement {
    int bvhNodeIdx; // idx of the BVH node
};

// was used for prototyping - commented because not needed
struct BvhHitRecord {
    bool hit; // was it hit
    float t; // time of hit
    vec3 n; // hit normal
    
    int bvhHits; // number of hits in the bvh structure - used for debugging
};


    
// called for processing the hit of a leaf node
void bvhProcessLeafHit(vec3 ro, vec3 rd, int leafNodeIdx, inout BvhHitRecord hitRecord) {
    { // check collision with leaf
        if (bvhLeafNodeType[leafNodeIdx] == 0) { // sphere
            vec4 positionAndRadius = bvhLeafNodeVertex0[leafNodeIdx];

            // shoot ray aganst sphere
            float thisIntersection = iSphere(ro, rd, positionAndRadius);
            if (thisIntersection >= 0.0) { // has hit
                if (!hitRecord.hit) { // is first hit
                    hitRecord.t = thisIntersection;
                    hitRecord.hit = true;
                    
                    vec3 p = ro + rd*thisIntersection;
                    hitRecord.n = (p - positionAndRadius.xyz) * (1.0 / positionAndRadius.w);
                }
                else if (thisIntersection < hitRecord.t) {
                    hitRecord.t = thisIntersection;
                    hitRecord.hit = true;
                    
                    vec3 p = ro + rd*thisIntersection;
                    hitRecord.n = (p - positionAndRadius.xyz) * (1.0 / positionAndRadius.w);
                }
            }
        }
        else if (bvhLeafNodeType[leafNodeIdx] == 1) { // polygon
            vec3 vertex0 = bvhLeafNodeVertex0[leafNodeIdx].xyz; 
            vec3 vertex1 = bvhLeafNodeVertex1[leafNodeIdx].xyz;
            vec3 vertex2 = bvhLeafNodeVertex2[leafNodeIdx].xyz;

            TriangleIntersection intersectionResult = iTriangle(
                ro, rd,
                vertex0,
                vertex1,
                vertex2
            );
            float thisIntersection = intersectionResult.t;

            if (thisIntersection > 0.0) {
                if (!hitRecord.hit) { // is first hit
                    hitRecord.t = thisIntersection;
                    hitRecord.hit = true;
                    
                    vec3 p = ro + rd*thisIntersection;
                    hitRecord.n = normalize(intersectionResult.n);
                }
                else if (thisIntersection < hitRecord.t) {
                    hitRecord.t = thisIntersection;
                    hitRecord.hit = true;
                    
                    vec3 p = ro + rd*thisIntersection;
                    hitRecord.n = normalize(intersectionResult.n);
                }
            }
        }
    }
}



// traveral of bvh
BvhHitRecord bvhTraverse(in vec3 ro, in vec3 rd) {
    // see https://devblogs.nvidia.com/thinking-parallel-part-ii-tree-traversal-gpu/
    // for details of iterative BVH traversal

    // see http://www.realtimerendering.com/raytracing/Ray%20Tracing_%20The%20Next%20Week.pdf
    //     page 14 for details on BVH traversal

    // TODO< further optimizations as suggested in the article https://devblogs.nvidia.com/thinking-parallel-part-ii-tree-traversal-gpu/ >

    BvhHitRecord hitRecord;
    hitRecord.hit = false;
    hitRecord.t = -1.0;
    hitRecord.bvhHits = 0;

    BvhStackElement stack[64];
    int stackIdx = 0;
    
    
    

    { // push root BVH node on stack
        stack[stackIdx++].bvhNodeIdx = bvhRootNodeIdx; // set top BVH element to element at index
                                          // we might want to revise this when we have multiple BVH's
    }



    for(;;){
        bool isStackEmpty = stackIdx <= 0;
        if (isStackEmpty) {
            break;
        }

        int currentBvhNodeIdx = stack[(stackIdx--)-1].bvhNodeIdx; // pop and read out
        BvhNodeInfo2 currentBvhNode = retBvhNodeAt(currentBvhNodeIdx);

        mat4 mat;
        mat = translate(-currentBvhNode.aabbCenter); // negate because we move the ray origin
        bool aabbHit = sBox(ro, rd, mat, currentBvhNode.aabbExtend) >= 0.0;
        
        
        bool insideAabb = testInsideAabb(ro, currentBvhNode.aabbCenter, currentBvhNode.aabbExtend);

        // if bounding box is hit or when we are inside then we process the node
        if (aabbHit || insideAabb) {
            // for debugging
            hitRecord.bvhHits+=1;

            if (currentBvhNode.isLeaf) {
                int leafNodeIdx = currentBvhNode.leafNodeIdx;
                
                bvhProcessLeafHit(ro, rd, leafNodeIdx, hitRecord);
            }
            else {
                { // push right
                    BvhStackElement toBePushed;
                    toBePushed.bvhNodeIdx = currentBvhNode.childrenRight;

                    // push
                    stack[stackIdx++] = toBePushed;
                }

                { // push left
                    BvhStackElement toBePushed;
                    toBePushed.bvhNodeIdx = currentBvhNode.childrenLeft;

                    // push
                    stack[stackIdx++] = toBePushed;
                }
            }
        }
    }

    return hitRecord;
}

*/












// scene sphere position and radius
/////const vec4 spherePosR[1] = vec4[1](vec4(0.0, 0.0, 0.0, 0.3));


// screen space indirection
// We are using a (virtual) 2d array of a grid to accelerate the rasterization/raytracing/raymarching.
// This is done by putting the objects which are visible in the given rectangle into the bucket.
// Addressing of this array is done with calcScreenSpaceIdx()
// This array points to the "objectArray" by a index.

const int screenSpaceInd2dArr[20] = int[20](
    -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, 
    -1,-1,-1,-1,-1, -1,-1,-1,-1,-1
);

// length of given bucket
// n objects are read starting from the index read from screenSpaceInd2dArr
const int screenSpaceInd2dBucketLengthArr[20] = int[20](
    -1,-1,-1,-1,-1, -1,-1,-1,-1,-1, 
    -1,-1,-1,-1,-1, -1,-1,-1,-1,-1
);

int calcScreenSpaceIdx(int bucketX, int bucketY) {
    return bucketX + bucketY * 10; // TODO< 10 has to be overwritten by a value computed from a high resolution and subdivision - for example 16 pixel and UHD
}


// arrays describing the objects

// object type
// 0 : rasterized sphere  NOT IMPLEMENTED
// 1 : raytraced sphere  NOT IMPLEMENTED
// 2 : raytraced triangle  NOT IMPLEMENTED
// 3 : raytraced   NOT IMPLEMENTED
// TODO< other types >

const int globalTypeArr[1] = int[1](-1);

// first vertex/position/radius encoding
const vec4 globalVec0EncodingArr[1] = vec4[1](vec4(0.0,0.0,0.0,0.0));

// material id
const int globalMaterialIdArr[1] = int[1](0);




// array describing the materials
const vec4 matBaseColorArr[1] = vec4[1](vec4(1.0,0.0,0.0,1.0));



// function to render a pixel with a given 
vec4 rayIntoBucket(int bucketX, int bucketY,  vec3 cameraP, vec3 rayDir) {
    vec3 color = vec3(0.0, 0.0, 0.0);
    
    int screenSpaceIndirectionIdx = calcScreenSpaceIdx(bucketX, bucketY);
    
    // we have to read a list of objects/BVH nodes/etc from the linear array
    // so we need to realize an indirection mechanism like pointers are doing it in CPU land
    int indirectionBucketStartIdx = screenSpaceInd2dArr[screenSpaceIndirectionIdx];
    int indirectionBucketLength = screenSpaceInd2dBucketLengthArr[screenSpaceIndirectionIdx];
    
    for(
        int iGlobalArrIdx=indirectionBucketStartIdx;
        iGlobalArrIdx<indirectionBucketStartIdx+indirectionBucketLength;
        iGlobalArrIdx++
    ) {
        // now we need to dererefence the
        // * primitive
        // * instancing primitive
        // * (top) BVH node
        // * rasterization node
        
        // TODO TODO TODO TODO< dereference and render object >
        
        // TODO TODO TODO TODO< keep track of z buffer and object id >
    }
    
    // TODO TODO TODO< do shading >
    
    
    return vec4(color, 1.0);

}

// do we want to test/render the rendering of simple volumetrics?
//#define RENDER_VOLUME0

// do we want to test-render the AABB test
//#define RENDERTEST_AABB0

// do we want to test-render the BVH test
#define RENDERTEST_BVH0

// raytracing of simple atmosphere
void mainImage2(out vec4 fragColor, in vec2 uv, in float screenRatio) {
    float iTime = 0.0; // HACK

    // remap
    vec2 uv11 = uv * vec2(2.0) - vec2(1.0);
    uv11.y *= (screenRatio); // screen ratio correction

    // pixel color
    vec3 col = vec3(0.0);
    
    
    vec3 pos0 = vec3(0.0,0.0,1.0);
    float radius0 = 0.3;
    float radius1 = 0.25;
    
    vec3 dir = normalize(vec3(uv11,1.0));
    vec3 cameraPos = vec3(0.0,0.0,0.0);
    
    
    
    for(int instanceI=0;instanceI<10;instanceI++) {
    	

        // NOTE< we move the camera - else we are inside the testpolygon! >
        //mat4 mat = mat4(1.0);
        mat4 mat = rotateY(iTime * 0.1 * float(instanceI));
        mat = translate(vec3(-7.0 + float(instanceI) * 0.2, 0.0, 10.0)) * mat;
        MeshIntersection meshIntersection = rayVsMesh(cameraPos, dir, 0, 4, mat);
        if (meshIntersection.intersectionT >= 0.0) {
            float light = dot(meshIntersection.triangle.n, vec3(0.0,0.0,-1.0));

            col = vec3(1.0, light, light); // indicate with the color that we had hit the mesh
        }
    }
    
    
#ifdef RENDER_VOLUME0
    vec4 sphere0PositionAndRadius = vec4(pos0,radius0);
    vec2 tSphere0 = iSphere2(cameraPos, dir, sphere0PositionAndRadius);
    
    vec4 sphere1PositionAndRadius = vec4(pos0,radius1);
    vec2 tSphere1 = iSphere2(cameraPos, dir, sphere1PositionAndRadius);
    
    if (tSphere0.x < 0.0) {
        //col = vec3(0.8, 0.8, 0.0);
    }
    else {
        float dist = tSphere0.y - tSphere0.x;
        
        if (tSphere1.x >= 0.0) {
            dist = tSphere1.x - tSphere0.x;
        }
        
        //col = vec3(dist*1.3);
        col += vec3(dist*0.6);
    }
#endif
    
    
    // sphere tracing of primary ray for test shape
    {
    	float iTime = 0.0; // HACK

        bool hit = false;
        vec3 n = vec3(0.0,0.0,0.0); // normal
        
        vec3 itP = cameraPos; // iteration position
        for(int iStep=0;iStep<50;iStep++) {
            float dist;
            dist = sdSphere(itP, vec3(0.0,0.0,3.0), 1.0);
            dist += 0.1*sdPertubeCos(itP, vec3(10.0,3.0,0.0), iTime);
            
            // check if hit
            if (dist < 0.01) {
                hit = true;
                break;
            }
            
            itP += (dir * dist); // advance
        }
        
        // compute normal
        if (hit) {
            // ref https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
            
            const float h = 0.001; // or some other value
            const vec2 k = vec2(1,-1);
            
            float dist;

            // TODO< remove this dependency on time and the test here >
            float iTime = 0.0;
            
            dist = sdSphere(itP, vec3(0.0,0.0,3.0)+k.xyy*h, 1.0);
            dist += 0.1*sdPertubeCos(itP, vec3(10.0,3.0,0.0)+k.xyy*h, iTime);
    
            float pnn = dist;
            
            dist = sdSphere(itP, vec3(0.0,0.0,3.0)+k.yyx*h, 1.0);
            dist += 0.1*sdPertubeCos(itP, vec3(10.0,3.0,0.0)+k.yyx*h, iTime);
    
            float nnp = dist;
            
            dist = sdSphere(itP, vec3(0.0,0.0,3.0)+k.yxy*h, 1.0);
            dist += 0.1*sdPertubeCos(itP, vec3(10.0,3.0,0.0)+k.yxy*h, iTime);
    
            float npn = dist;
            
            dist = sdSphere(itP, vec3(0.0,0.0,3.0)+k.xxx*h, 1.0);
            dist += 0.1*sdPertubeCos(itP, vec3(10.0,3.0,0.0)+k.xxx*h, iTime);
    
            float ppp = dist;
            
            // feed into function to compute normal
            n = sdNormalFast(pnn, nnp, npn, ppp);
            
        }
        
        if (hit) {
            vec3 lightDir = vec3(0.0,0.0,-1.0);
            
            // diffuse shading of implicit surface
            float diffuse = dot(lightDir, n);
            diffuse = max(0.0, diffuse);
            
            // compute reflected direction
            vec3 reflected = reflect(dir, n);
            
            // shoot ray against explicit sphere
            float sphereT = iSphere(itP, reflected, vec4(0.0,0.0,0.0,1.5));
            
            
            
            
                col = vec3(0.0,diffuse,0.0);
            
            vec3 secondaryColor = vec3(0.0);
            
            if (sphereT > 0.0) { // hit?
                secondaryColor = vec3(1.0, 0.0, 0.0);
            }
            
            col = col * 0.6 + secondaryColor * 0.4;
            
        }
    }
    

#ifdef RENDERTEST_AABB0
    {
        mat4 mat;
        mat = translate(-vec3(2.0,2.0,10.0));
        vec3 extend = vec3(1.0,1.0,1.0);
        bool aabbHit = sBox(cameraPos, dir, mat, extend) >= 0.0;

        if (aabbHit) {
            col = vec3(1.0, 1.0, 1.0);
        }
    }
#endif


#ifdef RENDERTEST_BVH0
    /* commented because BVH doesn't work because we need to use textures! 
    {
        vec3 rayOrigin = cameraPos;
        BvhHitRecord bvhHitRecord = bvhTraverse(rayOrigin, dir);

        // visualize number of bvh hits
        col += vec3(float(bvhHitRecord.bvhHits) / 3.0);
        
        if (bvhHitRecord.hit) {
            col = vec3(0.0, 1.0, 0.0); // visualize hit
        }
        
    }
    */
#endif

    // Output to screen
    fragColor = vec4(col,1.0);
    
}

// TODO< build complete version of encoding and render a bucket! >


// TODO< call into mainImage() from main() >























in VS_OUTPUT {
    vec3 Color;
} IN;

out vec4 Color;

uniform vec4 ourColor;

void main() {
    Color = vec4(IN.Color, 1.0f);
    //Color = ourColor;
}
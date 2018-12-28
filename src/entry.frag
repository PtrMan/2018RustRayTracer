
///////////////////////////////
// BVH







// we store the BVH in a SoA because it has to be accessible by the host
//  * store the index of the left children node, is -1 if invalid 
uniform int bvhNodeChildrenLeft[];

//  * store the index of the right children node, is -1 if invalid 
uniform int bvhNodeChildrenRight[];

//  * is the specific bvh node a leaf
uniform int bvhIsLeaf[];

uniform vec4 bvhAabbCenter[];

uniform vec4 bvhAabbExtend[];

// * indices to bvh leaf nodes in global array
uniform int bvhLeafNodeIndices[];

uniform int bvhRootNodeIdx; // root node is at index 0





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














in VS_OUTPUT {
    vec3 Color;
} IN;

out vec4 Color;

uniform vec4 ourColor;

void main() {
    Color = vec4(IN.Color, 1.0f);
    //Color = ourColor;
}
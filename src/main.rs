// hybrid of a rasterizer and ray marcher
//
// Combines a fast way to rasterize spheres with ray marching for implicit surfaces



// compile and run with (fast debug)
// cargo test && cargo build --release && cargo run --release
// compile and run with (dev)
// cargo test && cargo build && cargo run



// create video
// ffmpeg -y -pattern_type glob -framerate 25 -i "*.ppm" output.avi && rm ./*.ppm




// TODO< 1 dimensional texture for circular plane >


// TODO< use light struct and render lights >


// TODO< make camera parameters customizable >

// TODO< add visualization for debug crosses >


// TODO< compute planes of bounding box >
// TODO< test bounding box in renderer >
// TODO< transform object/rotated bounding box by matrix >
// TODO< put implicit surface in it and use inverse matrix for transforming into the space and back >



// TODO< finish unify camera ray computation for orthogonal shadow computation of    rasterization and raytracing/raymarching

// TODO< use analytical normal computation of implicit surface in renderer and test it visually >

// TODO< integrate raymarching codepath into rendering "pipeline" fully >



// TODO< add unittest for collision of ray with bounding box >
//       there are some easy cases where the ray should collide with it or not
//       for example in front or behind the edge, should collide in more of the half x position, etc



// TODO< add soft patricles with the formulas from iq >

#![allow(non_snake_case)]

mod opengl;

#[derive(Clone)]
pub struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}


impl Vec3 {
	pub fn new(x: f64, y: f64, z: f64) -> Vec3 {
		Vec3{x:x,y:y,z:z}
    }

    pub fn magnitude(&self) -> f64 {
    	dot(&self, &self).sqrt()
    }


	pub fn scale(&self, s: f64) -> Vec3 {
    	Vec3{x:self.x*s, y:self.y*s, z:self.z*s}
	}

    pub fn xy(&self) -> Vec2 {
    	Vec2{x:self.x, y:self.y}
    }
}

impl<'a, 'b> std::ops::Add<&'b Vec3> for &'a Vec3 {
    type Output = Vec3;

    fn add(self, rhs: &'b Vec3) -> Vec3 {
        Vec3{x:self.x + rhs.x, y:self.y + rhs.y, z:self.z + rhs.z}
    }
}

impl<'a, 'b> std::ops::Sub<&'b Vec3> for &'a Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: &'b Vec3) -> Vec3 {
        Vec3{x:self.x - rhs.x, y:self.y - rhs.y, z:self.z - rhs.z}
    }
}

pub fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a.x*b.x + a.y*b.y + a.z*b.z
}

pub fn normalize(v: &Vec3) -> Vec3 {
	let magnitude = v.magnitude();
	v.scale(1.0 / magnitude)
}

fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
	let a2 = Vector3::new(a.x, a.y, a.z);
	let b2 = Vector3::new(b.x, b.y, b.z);
	let crossResult = Vector3::from(a2.cross(&b2));
	Vec3::new(crossResult.x, crossResult.y, crossResult.z)
}






// calculates height of a  projected sphere
fn calcHeightOfSphereOnUnit(distUnit: f64) -> Option<f64> {
    if distUnit < 1.0 {
        // 1 == x*x + y*y
        // 1 - x*x = y*y
        // y = sqrt(1 - x*x)

        let sphere = (1.0 - distUnit*distUnit).sqrt();
        return Some(sphere);
    }
    None
}

#[derive(Clone)]
pub struct RasterizedSphere {
    pub relativeHeight: f64,

    pub id: i64,

    pub z: f64,
}

// information about the rasterized or raytraced surface
#[derive(Clone)]
pub enum PixelSurfaceInfo {
    RasterizedSphere(RasterizedSphere),
    RaytracedCirlcePlaneIntersection{ id: i64, rayT: f64},
    RaytracedCappedCylinderIntersection{id: i64, rayT: f64, n: Normal}
}



// compute the depth/z value of a pixel
fn calcDepth(scene: &Scene, pixelSurfaceInfo: &PixelSurfaceInfo) -> f64 {
	match pixelSurfaceInfo {
		PixelSurfaceInfo::RasterizedSphere(rasterizedSphere) => {
			let primitiveSphere = &scene.spherePrimitives[rasterizedSphere.id as usize];

            let rMulHeight = rasterizedSphere.relativeHeight * primitiveSphere.r;

            let depth = calcZValueOfProjectedSphere(rasterizedSphere.z, rMulHeight, EnumFace::FRONT);

            return depth;
		}
		PixelSurfaceInfo::RaytracedCirlcePlaneIntersection{id, rayT} => {
			return *rayT;
		}
		PixelSurfaceInfo::RaytracedCappedCylinderIntersection{id, rayT, n} => {
			return *rayT;
		}
	}
}


fn projectSphereAtZBuffer(projectedSphere: &ProjectedSphere, p: &Vec2, r: f64, yi: i64, xi: i64,  zBuffer: &Map2d<f64>) -> Option<RasterizedSphere> {
    let distanceToCenterUnit = projectedSphere.calcOthoDistanceByAbsPosition(&p);

    let heightOfSphere:Option<f64> = calcHeightOfSphereOnUnit(distanceToCenterUnit);

    match heightOfSphere {
        Some(relativeHeight) => {
            let zBufferValue = *zBuffer.retAtUnchecked(yi, xi);

            //  we use back-face culling as a trick against Shadow acne
            //  see http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
            let comparedDepth = calcZValueOfProjectedSphere(projectedSphere.z, relativeHeight*projectedSphere.r, EnumFace::BACK);

            Some(RasterizedSphere{relativeHeight:relativeHeight,  id:projectedSphere.id, z:projectedSphere.z})
        }
        None => {
            None
        }
    }
}


// a sphere which is projected on the plane of the camera
pub struct ProjectedSphere {
    pub id: i64,
    pub z: f64,
    pub center: Vec2,
    
    pub r: f64, // radius of unprojected sphere

    pub axisA: Vec2,
    pub axisB: Vec2,
}

impl ProjectedSphere {
	// computes the relative distance from 0.0 to 1.0 or above
	// result 0.0 means that it in the center, 1.0 at the outside and greater than 1.0 that it is not in the ellipse
	// /param p position of sampled position
	fn calcOthoDistanceByAbsPosition(&self, p: &Vec2) -> f64 {
		let rel = p - &self.center;
		calcEllipseDistToCenter(&rel, &self.axisA, &self.axisB)
	}
}





#[derive(Clone, Copy)]
pub enum EnumFace {
    FRONT,
    BACK
}

// computes the projected depth of a projected sphere
// /param z the z value of the projected center of the sphere
// /param absoluteProjectedRadius the 
// /param face used projected face
//        back-face rendering for a method against shadow acne
// /return projected depth
fn calcZValueOfProjectedSphere(z: f64, absoluteProjectedRadius: f64, face: EnumFace) -> f64 {
    match face {
        EnumFace::FRONT => {
            z - absoluteProjectedRadius //z - absoluteProjectedRadius
        }

        EnumFace::BACK => {
            z + absoluteProjectedRadius
        }
    }
}


pub fn rasterizeSphere(projectedSphere: &ProjectedSphere, r: f64, zBuffer: &mut Map2d<f64>, resBuffer: &mut Vec<Option<PixelSurfaceInfo>>, face: EnumFace) {
    // screen space bounding box
    let mut minX;
    let mut maxX;
    let mut minY;
    let mut maxY;

    {
    	// helper to convert from relative screen coordinates to absolute pixels
    	fn convRelativeToAbsoluteAsInt(relative: f64, resolution: i64,   aspectRatio: f64) -> i64 {
    		let resolutionAsReal = resolution as f64;

    		let relativeIn01 = (relative + 1.0) * 0.5; // relative in range [0.0;1.0)

    		let mappedAsInt = relativeIn01 * aspectRatio * resolutionAsReal;
    		mappedAsInt as i64
    	}

    	// TODO< compute length by length of vector spanned by the axis >
    	let mut maxAxisLength = projectedSphere.axisA.magnitude();
    	maxAxisLength = maxAxisLength.max(projectedSphere.axisB.magnitude());


    	let maxAxisLengthAsInt = maxAxisLength as i64;

    	// we want to render the whole image for the sphere in the worst case
    	minX = 0;
    	maxX = zBuffer.retWidth();
    	minY = 0;
    	maxY = zBuffer.retHeight();

    	// maximal bounding box

    	let mut boundXMin = convRelativeToAbsoluteAsInt(projectedSphere.center.x - maxAxisLength, zBuffer.retWidth(), 1.0) + 1;
    	boundXMin -= 1; // one because we need to include border
    	let mut boundXMax = convRelativeToAbsoluteAsInt(projectedSphere.center.x + maxAxisLength, zBuffer.retWidth(), 1.0) + 1;
    	boundXMax += 1; // one because we need to include border


    	minX = minX.max(boundXMin);
    	maxX = maxX.min(boundXMax);

    	// TODO< division for ratio may be wrong >

    	let mut boundYMin = convRelativeToAbsoluteAsInt(projectedSphere.center.y - maxAxisLength, zBuffer.retHeight(), zBuffer.retHeight() as f64 / zBuffer.retWidth() as f64) + 1;
    	boundYMin -= 1; // one because we need to include border
    	let mut boundYMax = convRelativeToAbsoluteAsInt(projectedSphere.center.y + maxAxisLength, zBuffer.retHeight(), zBuffer.retHeight() as f64 / zBuffer.retWidth() as f64) + 1;
    	boundYMax += 1; // one because we need to include border

    	minY = minY.max(boundYMin);
    	maxY = maxY.min(boundYMax);
    }

    //println!("{} {}", minX, maxX);

    for yi in minY..maxY {
        for xi in minX..maxX {
            let x = ((xi as f64) / 512.0) * 2.0 - 1.0;
            let y = ((yi as f64) / 512.0) * 2.0 - 1.0;

            let p = Vec2::new(x, y);

            let rasterizeResult: Option<RasterizedSphere> =
                projectSphereAtZBuffer(&projectedSphere, &p, r,   yi, xi,  zBuffer);
            
            match &rasterizeResult {
                &Some(ref v) => {
                    let depth = calcZValueOfProjectedSphere(v.z, v.relativeHeight * r, face);

                    if depth < *zBuffer.retAtUnchecked(yi, xi) {
	                    // OPTIMIZATION< maybe refcouting is faster here instead of cloning >
	                    resBuffer[(yi * 512 + xi) as usize] = Some(PixelSurfaceInfo::RasterizedSphere(v.clone()));

	                    // write z buffer
	                	zBuffer.setAtUnchecked(yi, xi, &depth);
                    }

                }
                &None => {}
            }
        }
    }
}


// all shading information
#[derive(Clone)]
struct Shading {
	colorR: f64,
    colorG: f64,
    colorB: f64,
}

// contains all the informations of the primitive in world space!
struct PrimitiveSphere {
    id: i64,
    shading: Shading,

    pos: Point,
    r: f64,
}

impl HasCenter for PrimitiveSphere {
	fn retCenter(&self) -> Point {
		self.pos.clone()
	}
}

impl HasAabb for PrimitiveSphere {
	fn retAabbExtend(&self) -> Vec3 {
		Vec3::new(self.r*2.0, self.r*2.0, self.r*2.0)
	}
	//fn setAabbExtend(&mut self, extend: &Vec3);

	fn retAabbCenter(&self) -> Point {
		self.pos.clone()
	}
}



struct PrimitiveCirclePlane {
	id: i64,
	shading: Shading,

	radius: f64,

	// position is a point on the plane and normal is the normal of it
	pos: Point,
	n: Normal,

}

struct PrimitiveCappedCylinder {
	id: i64,
	shading: Shading,

	pA: Point,
	pB: Point,

	radiusA: f64,
	radiusB: f64,

}

struct Light {
	position: Point,

	color: Color32,
	maxRadius: f64, // maximal render radius - can be infinite

	brightness: f64,
}

// scene description
struct Scene {
	spherePrimitives: Vec<PrimitiveSphere>,
	circlePlanePrimitives: Vec<PrimitiveCirclePlane>,
	cappedCylinderPrimitives: Vec<PrimitiveCappedCylinder>,

	lights: Vec<Light>,
}

impl Scene {
	fn new() -> Scene {
		Scene {
			spherePrimitives: Vec::new(),
			circlePlanePrimitives: Vec::new(),
			cappedCylinderPrimitives: Vec::new(),

			lights: Vec::new(),
		}
	}
}


// contain all viewport related information
struct Viewport {
    // rasterized buffer
    rasterized: Vec<Option<PixelSurfaceInfo>>,

    zBuffer: Map2d<f64>,

    // do we need to shade it?
    enableShading: bool,

    // which face of the bodies should be rendered?
    face: EnumFace,

    camera: Camera,
}

extern crate time;

impl Viewport {
	// rasterizes all rasterizable primitives
    fn rasterize(&mut self, scene: &Scene) {
        let mut projectedSpheres: Vec<ProjectedSphere> = Vec::new();

        // (*) project spheres
        for iSphere in scene.spherePrimitives.iter() { // iterate over all spheres and project them
        	let z = self.camera.calcDepthOfProjectedPoint(&iSphere.pos);

        	match self.camera.type_ {
        		EnumCameraType::ORTHOGONAL => {

		            let projectedPosition = self.camera.project(&iSphere.pos.p);

		            ///println!("# p-pos=<{},{},{}>", projectedPosition.x, projectedPosition.y, projectedPosition.z);

		        
		            projectedSpheres.push(ProjectedSphere{
		                id: iSphere.id,
		                z: z, // commented because it worked - but it is equivalent   projectedPosition.z,
		            
		                center: Vec2::new(projectedPosition.x, projectedPosition.y),

		                r: iSphere.r,
		            	
		                axisA: Vec2::new(iSphere.r, 0.0),
		                axisB: Vec2::new(0.0, iSphere.r),
		            });
        		}

        		EnumCameraType::PERSPECTIVE => {


        			let mut sphereWorldPosition: Vec3 = iSphere.pos.p.clone();
					//let sphereRadius = 0.5;
					
					// transfrom from world to camera local coordinate system
					//  we need to have the matrix to transform a position from world to camera space
					//   we do this by calculating the transformaton of the space spanned by the three vectors
					let relativeGlobalToCamera:Matrix44 = Matrix44::new(
						self.camera.sideNormalized.x, self.camera.sideNormalized.y, self.camera.sideNormalized.z, 0.0,
						self.camera.upNormalized.x, self.camera.upNormalized.y, self.camera.upNormalized.z, 0.0,
						self.camera.dirNormalized.v.x, self.camera.dirNormalized.v.y, self.camera.dirNormalized.v.z, 0.0,
						0.0, 0.0, 0.0, 1.0
					);


					let mut sphereLocalToCameraPosition = mul(&relativeGlobalToCamera, &(&sphereWorldPosition - &self.camera.position.p));
					
					let fov = 3.14 / 2.0;
					
					let cameraMat;
					{ // compute camera matrix
						let perspective = Perspective3::new((512 as f64) / (512 as f64), fov, 0.1, 1000.0);

			    		// convert to a `Matrix4`
						let perspectiveMat = perspective.to_homogeneous();


						// construct matrix, we need to invert the z axis because the projected camera looks into negative direction, while we assume a positive direction in the engine
						let reflectionZMat = Matrix44::new_nonuniform_scaling(&Vector3::new(1.0, 1.0, -1.0));

						// we need to combine it with the matrix to reflect the Z axis
						cameraMat = perspectiveMat * reflectionZMat;
					}

					let projectionResult:ProjectionResult;
					{ // compute projection result
						let sphereEncodedAsVec4 = Vec4::new(sphereLocalToCameraPosition.x, sphereLocalToCameraPosition.y, sphereLocalToCameraPosition.z, iSphere.r);
						
						projectionResult = projectSphere(&sphereEncodedAsVec4, &cameraMat, fov);
					}


					projectedSpheres.push(ProjectedSphere{
		                id: iSphere.id,
		                z: z,
		            
		                center: projectionResult.center,

		                r: iSphere.r,
		            	
		                axisA: projectionResult.axisA,
		                axisB: projectionResult.axisB,
		            });


        		}
        	}


        }


        let timeStartInNs = time::precise_time_ns();

        // (*) rasterize spheres
        for iProjectedSphere in projectedSpheres.iter() {
            rasterizeSphere(&iProjectedSphere, iProjectedSphere.r,    &mut self.zBuffer, &mut self.rasterized, self.face);
        }

        let timeEndInNs = time::precise_time_ns();

        println!("rasterization took t={}us", (timeEndInNs - timeStartInNs) / 1000);
    }


    // TODO< make private and call for main rendering >
    fn processRaymarchingRays(&mut self) {

    	for yi in 0..512 {
        	for xi in 0..512 {
    			let rayDepthOption:Option<f64> = rayEntry_ShadowRay_testing(&self.camera, xi, yi);

    			if rayDepthOption.is_some() {
    				let rayDepth = rayDepthOption.unwrap();
    				if rayDepth < *self.zBuffer.retAtUnchecked(yi, xi) {
    					self.zBuffer.setAtUnchecked(yi, xi, &rayDepth);
    				}
    			}
    		}
    	}
    }

    // TODO< make private and call from main rendering >

    fn processRaytracingRays(&mut self, scene: &Scene) {
    	for yi in 0..512 {
        	for xi in 0..512 {
        		let rayOriginAndDirection = self.camera.calcRayOriginAndDirection(xi, yi);
                let (rayOriginPosition, rayOriginDirection) = rayOriginAndDirection;

                for iPlane in scene.circlePlanePrimitives.iter() {
                	// translate to plane struct
                	let plane = Plane{
                		n: iPlane.n.clone(),
                		center: iPlane.pos.clone()
                	};

                	// compute intersection
                	let intersectionTOption:Option<f64> = calcRayPlane(&rayOriginPosition, &rayOriginDirection, &plane);// compute the 
                	if intersectionTOption.is_none() {
                		continue;
                	}

                	let intersectionT = intersectionTOption.unwrap();

                	if intersectionT < 0.0 {
                		continue; // behind the camera
                	}

            		if intersectionT > *self.zBuffer.retAtUnchecked(yi, xi) {
            			continue; // behind a known intersection with another object
            		}

                	let intersectionP = &rayOriginPosition.p + &rayOriginDirection.v.scale(intersectionT);

                	// check if it is in the radius
                	let distance = (&intersectionP - &iPlane.pos.p).magnitude();
                	if distance > iPlane.radius {
                		continue;
                	}

                	// it was a hit

    				let pixelSurfaceInfo = PixelSurfaceInfo::RaytracedCirlcePlaneIntersection{id:iPlane.id, rayT: intersectionT};
                	self.rasterized[(yi * 512 + xi) as usize] = Some(pixelSurfaceInfo);
                	self.zBuffer.setAtUnchecked(yi, xi, &intersectionT);
                }

                for iCappedCylinder in scene.cappedCylinderPrimitives.iter() {

                	let intersectionTAndNormal:Vec4 = iCappedCone(
                		&rayOriginPosition.p, &rayOriginDirection.v, 
               			&iCappedCylinder.pA.p, &iCappedCylinder.pB.p,
               			iCappedCylinder.radiusA, iCappedCylinder.radiusB
               		);
                	let intersectionT = intersectionTAndNormal.x;
                	let intersectionNormal = Vec3::new(intersectionTAndNormal.y, intersectionTAndNormal.z, intersectionTAndNormal.w);


                	if intersectionT < 0.0 {
                		continue; // behind the camera
                	}

            		if intersectionT > *self.zBuffer.retAtUnchecked(yi, xi) {
            			continue; // behind a known intersection with another object
            		}

                	// it was a hit

    				let pixelSurfaceInfo = PixelSurfaceInfo::RaytracedCappedCylinderIntersection{id:iCappedCylinder.id, rayT: intersectionT, n: Normal{v:intersectionNormal}};
                	self.rasterized[(yi * 512 + xi) as usize] = Some(pixelSurfaceInfo);
                	self.zBuffer.setAtUnchecked(yi, xi, &intersectionT);
                }
        	}
        }


    }
}


#[derive(Clone)]
struct Color32 {
    r: f32,
    g: f32,
    b: f32,
}


use std::fs::File;
use std::io::{Write, BufReader, BufRead};



// image is linear space colored image
fn writeColorImage(image: &Map2d<Color32>, path: &String) {



    let mut data = String::from("");
    { // serialize image
        data += &String::from("P3\n");
        data += &format!("{} {}\n", image.retWidth(), image.retHeight());
        data += &String::from("255\n");

        for iy in 0..image.retHeight() {
            for ix in 0..image.retWidth() {
                let color = image.retAtUnchecked(iy, ix);

                let mut r = color.r.max(0.0).min(1.0);
                let mut g = color.g.max(0.0).min(1.0);
                let mut b = color.b.max(0.0).min(1.0);

                // gamma correction
                let gamma = 2.2f32;
                r = f32::powf(r, 1.0f32/gamma);
                g = f32::powf(g, 1.0f32/gamma);
                b = f32::powf(b, 1.0f32/gamma);

                data += &format!("{} {} {}  ", (r * 255.0) as i64, (g * 255.0) as i64, (b * 255.0) as i64);            
            }

            data = data + "\n";
        }
    }

    { // write image
        let mut output = File::create(path);

        match output {
            Ok(mut m) => {
                write!(m, "{}", data.as_str());

            }
            Err(..) => {
                panic!("");
            }
        }
    }
}




// renders the color image of a scene
fn renderColorImage(scene: &Scene, viewport: &Viewport, viewportShadowmapping: &Viewport) -> Map2d<Color32> {
    let mut image: Map2d<Color32> = Map2d::<Color32>::new(512, 512, Color32{r:0.0,g:1.0,b:0.2});


    for iy in 0..512 {
        for ix in 0..512 {
            let iPixel:&Option<PixelSurfaceInfo> = &viewport.rasterized[(iy * 512 + ix) as usize];

            let mut r = 0.0;
            let mut g = 0.2;
            let mut b = 0.2;

            if iPixel.is_some() {
            	// direction of view, is the vector from the collision point to the camera "eye"
		        let viewDir = viewport.camera.retViewDirOfPixel(ix, iy);

		        let worldPosition;
		        let normal;

		        let shading;


                match iPixel {
                    &Some(ref pixelSurfaceInfo) => {
                    	match pixelSurfaceInfo {
							PixelSurfaceInfo::RasterizedSphere(rasterizedSphere) => {

		                        { // compute world position and normal of fragment
		                            let depth = calcDepth(&scene, &pixelSurfaceInfo); 


		                            let rayOriginAndDirection = viewport.camera.calcRayOriginAndDirection(ix, iy);
		                            let (rayOriginPosition, rayOriginDirection) = rayOriginAndDirection;

		                            let mut worldPosition2 = rayOriginPosition.p;
		                            worldPosition2 = &worldPosition2 + &rayOriginDirection.v.scale(depth);
		                            worldPosition = worldPosition2;

		                            let primitiveSphere = &scene.spherePrimitives[rasterizedSphere.id as usize];

		                        	let diffOfPositionToCenter = &worldPosition - &primitiveSphere.pos.p;

		                        	normal = Normal{v:diffOfPositionToCenter.scale(1.0/primitiveSphere.r)};

		                        	//println!("n=<{},{},{}>", normal.v.x, normal.v.y, normal.v.z);
		                        }


		                        let primitiveSphere = &scene.spherePrimitives[rasterizedSphere.id as usize];
		                        shading = primitiveSphere.shading.clone();
							}
							PixelSurfaceInfo::RaytracedCirlcePlaneIntersection{id, rayT} => {
								{ // calculate world position
									let depth = calcDepth(&scene, &pixelSurfaceInfo);

									let rayOriginAndDirection = viewport.camera.calcRayOriginAndDirection(ix, iy);
		                            let (rayOriginPosition, rayOriginDirection) = rayOriginAndDirection;

									let mut worldPosition2 = rayOriginPosition.p;
		                            worldPosition2 = &worldPosition2 + &rayOriginDirection.v.scale(depth);
		                            worldPosition = worldPosition2;
								}

								let primitivePlane = &scene.circlePlanePrimitives[*id as usize];
								shading = primitivePlane.shading.clone();

								normal = primitivePlane.n.clone();
							}

							PixelSurfaceInfo::RaytracedCappedCylinderIntersection{id, rayT, n} => {
								{ // calculate world position
									let depth = calcDepth(&scene, &pixelSurfaceInfo);

									let rayOriginAndDirection = viewport.camera.calcRayOriginAndDirection(ix, iy);
		                            let (rayOriginPosition, rayOriginDirection) = rayOriginAndDirection;

									let mut worldPosition2 = rayOriginPosition.p;
		                            worldPosition2 = &worldPosition2 + &rayOriginDirection.v.scale(depth);
		                            worldPosition = worldPosition2;
								}

								let primitivePlane = &scene.cappedCylinderPrimitives[*id as usize];
								shading = primitivePlane.shading.clone();

								normal = n.clone();
							}

						}

                    }
                    &None => {
                    	continue;
                    }
                }

                /////////////
                // shading



                let incommingLightDir = Vec3::new(1.0, 0.0, 0.0);

                // light direction is negative because we look from the surface
                let invertedIncommingLightDir = incommingLightDir.scale(-1.0);



                let mut diffuse = dot(&normal.v, &invertedIncommingLightDir);
                diffuse = diffuse.max(0.0);

                // see https://learnopengl.com/Lighting/Basic-Lighting
                let reflectionDir = reflect(&invertedIncommingLightDir, &normal);
                let specularMagnitude = dot(&viewDir.v, &reflectionDir.v).max(0.0).powi(32);

                let debugNoShading = false; // shading mode of no shading - for debugging other parts of the rendering
                if debugNoShading {
                	diffuse = 1.0; // overwrite for testing
                }

                let mut lightMagnitude = 1.0;


                
                let shadowMappingInLight;
                { // shadow mapping
                    //   project the position to the projection of the shadow mapping map
                    let projectedPosition = viewportShadowmapping.camera.project(&worldPosition);

                    //   read out depth from shadow map
                    //     we need the x and y position in texture space
                    let texX = (512/2) + (projectedPosition.x * ((512/2) as f64)) as i64;
                    let texY = (512/2) + (projectedPosition.y * ((512/2) as f64)) as i64;

                    //println!("projectedPosition=<{},{},{}>", projectedPosition.x, projectedPosition.y, projectedPosition.z);
                    //println!("pixel-idx=<{}, {}>", ix, iy);
                    //println!("shadowmap-idx=<{}, {}>", texX, texY);

                    let mut depthFromShadowMap = std::f64::INFINITY;
                    if viewportShadowmapping.zBuffer.isInBounds(texY, texX) {
                        depthFromShadowMap = *viewportShadowmapping.zBuffer.retAtUnchecked(texY, texX);
                    }

                    //println!("zbuffer={} projZ={}", depthFromShadowMap, projectedPosition.z);

                    //   compare the depth
                    let bias = 0.0; // set to positive values against shadow acne
                    shadowMappingInLight = depthFromShadowMap > projectedPosition.z - bias;
                }

                let shadowMappingInShadow = !shadowMappingInLight;

                let mut enableShadowMapping = true;

                if shadowMappingInShadow && enableShadowMapping {
                    lightMagnitude = 0.25; // ambient value

                }





                r = (diffuse + specularMagnitude) * lightMagnitude * shading.colorR;
                g = (diffuse + specularMagnitude) * lightMagnitude * shading.colorG;
                b = (diffuse + specularMagnitude) * lightMagnitude * shading.colorB;

                let debugDepthBuffer = false;
                if debugDepthBuffer {
                	let depthOfFragment = viewport.zBuffer.retAtUnchecked(iy, ix);

                	println!("{}", depthOfFragment);

                	// debug the depth for a sphere which is on x=0.0 and y=0.0
                	r = (*depthOfFragment * 0.5);
                	g = r;
                	b = r;
                }

                let debugNormal = false;
                if debugNormal {
                	r = normal.v.x.abs();
                	g = normal.v.y.abs();
                	b = normal.v.z.abs();
                }
            }

            let fragmentColor = Color32{r:r as f32, g:g as f32, b:b as f32};
            image.setAtUnchecked(iy, ix, &fragmentColor);
        }
    }

    return image;
}







fn testscene_raytracingPlane0() {
	let mut scene = Scene::new();

	let mut normal = Vec3::new(-1.0, 0.0, 1.0);
	normal = normalize(&normal);

	/*
	scene.circlePlanePrimitives.push(PrimitiveCirclePlane{
    	shading: Shading{
	        colorR: 0.02,
	        colorG: 0.02,
	        colorB: 1.0,
    	},

        id:0,
        pos:Point{p:Vec3::new(0.01,0.01,2.0)},

        radius: 0.5,
        n:Normal{v:normal},
	});
	*/

	scene.cappedCylinderPrimitives.push(PrimitiveCappedCylinder {
		id: 0,
		shading: Shading{
	        colorR: 1.0,
	        colorG: 0.02,
	        colorB: 0.02,
    	},

		pA: Point::new(0.01, 10.01, 10.01),
		pB: Point::new(0.01, 0.01, 10.01),

		radiusA: 0.3,
		radiusB: 0.2,
	});




    // viewpor for shadow mapping
    let mut viewport1: Viewport = Viewport{
        rasterized: vec![None; 512*512],

        zBuffer: Map2d::<f64>::new(512, 512, std::f64::INFINITY),

        enableShading: false, // don't enable shading because we are only interested in the depth channel
        
        //  we use back-face culling as a trick against Shadow acne
        //  see http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
        face: EnumFace::BACK,




        // the sun is shining in the z direction
        // this is the case because it simplifies the code - we don't need a projection for testing for now



        camera: Camera {
        	type_: EnumCameraType::ORTHOGONAL,

			position: Point{p:Vec3{x:0.0, y:0.0, z:0.0}}, // "base" position of camera
			dirNormalized: Normal{v: Vec3{x:0.0, y:0.0, z:1.0}}, // normalized direction

			upNormalized: Vec3{x:0.0, y:1.0, z:0.0}, // up vector
			sideNormalized: Vec3{x:1.0, y:0.0, z:0.0}, // side vector

			resolutionX: 512,
			resolutionY: 512,
		},

    };


    { // shadow mapping pass
        //viewport1.rasterize(&primitiveSpheres);

        //viewport1.processRaymarchingRays();
    }




    








    
    let mut viewport0: Viewport = Viewport{
        rasterized: vec![None; 512*512],

        zBuffer: Map2d::<f64>::new(512, 512, std::f64::INFINITY),

        // we need to render the usual front face for usual visualization
        face: EnumFace::FRONT,

        enableShading: true,


        camera: Camera {
        	type_: EnumCameraType::PERSPECTIVE,

			position: Point{p:Vec3{x:0.0, y:0.0, z:0.0}}, // "base" position of camera
			dirNormalized: Normal{v: Vec3{x:0.0, y:0.0, z:1.0}}, // normalized direction

			upNormalized: Vec3{x:0.0, y:1.0, z:0.0}, // up vector
			sideNormalized: Vec3{x:1.0, y:0.0, z:0.0}, // side vector

			resolutionX: 512,
			resolutionY: 512,
		},

    };



    { // normal rendering
        viewport0.rasterize(&scene);
        viewport0.processRaytracingRays(&scene);
    }


    // render and write out result image

    // (*) render color image
    let image: Map2d<Color32> = renderColorImage(&scene, &viewport0, &viewport1);

    // (*) write image
    writeColorImage(&image, &format!("img{:06}.ppm", 0));
}










// test scene for perspective camera

fn testscene_perspectiveSimple1() {
	let mut primitiveSpheres: Vec<PrimitiveSphere> = Vec::new();

    primitiveSpheres = Vec::new();


    primitiveSpheres.push(PrimitiveSphere{
    	shading: Shading{
	        colorR: 0.02,
	        colorG: 0.02,
	        colorB: 1.0,
    	},

        id:0,
        pos:Point{p:Vec3{x:0.01,y:0.01,z:4.0}},
        r:0.5,
    });

    let mut scene = Scene::new();
    scene.spherePrimitives = primitiveSpheres;



    // viewpor for shadow mapping
    let mut viewport1: Viewport = Viewport{
        rasterized: vec![None; 512*512],

        zBuffer: Map2d::<f64>::new(512, 512, std::f64::INFINITY),

        enableShading: false, // don't enable shading because we are only interested in the depth channel
        
        //  we use back-face culling as a trick against Shadow acne
        //  see http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
        face: EnumFace::BACK,




        // the sun is shining in the z direction
        // this is the case because it simplifies the code - we don't need a projection for testing for now



        camera: Camera {
        	type_: EnumCameraType::ORTHOGONAL,

			position: Point{p:Vec3{x:0.0, y:0.0, z:0.0}}, // "base" position of camera
			dirNormalized: Normal{v: Vec3{x:0.0, y:0.0, z:1.0}}, // normalized direction

			upNormalized: Vec3{x:0.0, y:1.0, z:0.0}, // up vector
			sideNormalized: Vec3{x:1.0, y:0.0, z:0.0}, // side vector

			resolutionX: 512,
			resolutionY: 512,
		},

    };


    { // shadow mapping pass
        //viewport1.rasterize(&primitiveSpheres);

        //viewport1.processRaymarchingRays();
    }




    








    
    let mut viewport0: Viewport = Viewport{
        rasterized: vec![None; 512*512],

        zBuffer: Map2d::<f64>::new(512, 512, std::f64::INFINITY),

        // we need to render the usual front face for usual visualization
        face: EnumFace::FRONT,

        enableShading: true,


        camera: Camera {
        	type_: EnumCameraType::PERSPECTIVE,

			position: Point{p:Vec3{x:0.0, y:0.0, z:0.0}}, // "base" position of camera
			dirNormalized: Normal{v: Vec3{x:0.0, y:0.0, z:1.0}}, // normalized direction

			upNormalized: Vec3{x:0.0, y:1.0, z:0.0}, // up vector
			sideNormalized: Vec3{x:1.0, y:0.0, z:0.0}, // side vector

			resolutionX: 512,
			resolutionY: 512,
		},

    };



    { // normal rendering
        viewport0.rasterize(&scene);
    }


    // render and write out result image

    // (*) render color image
    let image: Map2d<Color32> = renderColorImage(&scene, &viewport0, &viewport1);

    // (*) write image
    writeColorImage(&image, &format!("img{:06}.ppm", 0));
}




// more complicated test scene for perspective camera

fn testscene_perspectiveSimple2() {
	let mut primitiveSpheres: Vec<PrimitiveSphere> = Vec::new();

    primitiveSpheres = Vec::new();

    // multiple spheres
    primitiveSpheres.push(PrimitiveSphere{
    	shading: Shading{
		    
	        colorR: 0.02,
	        colorG: 0.02,
	        colorB: 1.0,
	    },

        id:0,
        pos:Point{p:Vec3{x:0.01,y:0.01,z:4.0}},
        r:0.5,
    });

    primitiveSpheres.push(PrimitiveSphere{
    	shading: Shading{
	    
        	colorR: 0.02,
        	colorG: 1.0,
        	colorB: 0.02,
        },

        id:1,
        pos:Point{p:Vec3{x:0.01,y:0.01,z:6.0}},
        r:0.5,
    });

    let mut scene = Scene::new();
    scene.spherePrimitives = primitiveSpheres;



    // viewpor for shadow mapping
    let mut viewport1: Viewport = Viewport{
        rasterized: vec![None; 512*512],

        zBuffer: Map2d::<f64>::new(512, 512, std::f64::INFINITY),

        enableShading: false, // don't enable shading because we are only interested in the depth channel
        
        //  we use back-face culling as a trick against Shadow acne
        //  see http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
        face: EnumFace::BACK,




        // the sun is shining in the z direction
        // this is the case because it simplifies the code - we don't need a projection for testing for now



        camera: Camera {
        	type_: EnumCameraType::ORTHOGONAL,

			position: Point{p:Vec3{x:0.0, y:0.0, z:0.0}}, // "base" position of camera
			dirNormalized: Normal{v: Vec3{x:0.0, y:0.0, z:1.0}}, // normalized direction

			upNormalized: Vec3{x:0.0, y:1.0, z:0.0}, // up vector
			sideNormalized: Vec3{x:1.0, y:0.0, z:0.0}, // side vector

			resolutionX: 512,
			resolutionY: 512,
		},

    };


    { // shadow mapping pass
        //viewport1.rasterize(&primitiveSpheres);

        //viewport1.processRaymarchingRays();
    }




    








    
    let mut viewport0: Viewport = Viewport{
        rasterized: vec![None; 512*512],

        zBuffer: Map2d::<f64>::new(512, 512, std::f64::INFINITY),

        // we need to render the usual front face for usual visualization
        face: EnumFace::FRONT,

        enableShading: true,


        camera: Camera {
        	type_: EnumCameraType::PERSPECTIVE,

			position: Point{p:Vec3{x:0.0, y:1.5, z:0.0}}, // "base" position of camera
			dirNormalized: Normal{v: Vec3{x:0.0, y:0.0, z:1.0}}, // normalized direction

			upNormalized: Vec3{x:0.0, y:1.0, z:0.0}, // up vector
			sideNormalized: Vec3{x:1.0, y:0.0, z:0.0}, // side vector

			resolutionX: 512,
			resolutionY: 512,
		},

    };



    { // normal rendering
        viewport0.rasterize(&scene);
    }


    // render and write out result image

    // (*) render color image
    let image: Map2d<Color32> = renderColorImage(&scene, &viewport0, &viewport1);

    // (*) write image
    writeColorImage(&image, &format!("img{:06}.ppm", 0));
}


// test scene of showing overlapping spheres
// one of the scenes used to test if the z computations, z depth test and z update are correctly done

fn testscene_overlappingSpheres() {
	let mut primitiveSpheres: Vec<PrimitiveSphere> = Vec::new();

    primitiveSpheres = Vec::new();


    primitiveSpheres.push(PrimitiveSphere{
    	shading: Shading{
	    
        	colorR: 0.02,
        	colorG: 0.02,
        	colorB: 1.0,
        },

        id:0,
        pos:Point{p:Vec3{x:0.0,y:0.0,z:0.0}},
        r:0.5,
    });


    primitiveSpheres.push(PrimitiveSphere{
    	shading: Shading{
	    
        	colorR: 1.0,
        	colorG: 1.0,
        	colorB: 1.0,
        },

        id:1,
        pos:Point{p:Vec3{x:0.0,y:0.0,z:0.5}}, // so it overlaps with the other sphere partially
        r:0.5,
    });

    let mut scene = Scene::new();
    scene.spherePrimitives = primitiveSpheres;






    // viewpor for shadow mapping
    let mut viewport1: Viewport = Viewport{
        rasterized: vec![None; 512*512],

        zBuffer: Map2d::<f64>::new(512, 512, std::f64::INFINITY),

        enableShading: false, // don't enable shading because we are only interested in the depth channel
        
        //  we use back-face culling as a trick against Shadow acne
        //  see http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
        face: EnumFace::BACK,




        // the sun is shining in the z direction
        // this is the case because it simplifies the code - we don't need a projection for testing for now



        camera: Camera {
        	type_: EnumCameraType::ORTHOGONAL,

			position: Point{p:Vec3{x:0.0, y:0.0, z:0.0}}, // "base" position of camera
			dirNormalized: Normal{v: Vec3{x:0.0, y:0.0, z:1.0}}, // normalized direction

			upNormalized: Vec3{x:0.0, y:1.0, z:0.0}, // up vector
			sideNormalized: Vec3{x:1.0, y:0.0, z:0.0}, // side vector

			resolutionX: 512,
			resolutionY: 512,
		},

    };


    { // shadow mapping pass
        //viewport1.rasterize(&primitiveSpheres);

        //viewport1.processRaymarchingRays();
    }




    








    
    let mut viewport0: Viewport = Viewport{
        rasterized: vec![None; 512*512],

        zBuffer: Map2d::<f64>::new(512, 512, std::f64::INFINITY),

        // we need to render the usual front face for usual visualization
        face: EnumFace::FRONT,

        enableShading: true,


        camera: Camera {
        	type_: EnumCameraType::ORTHOGONAL,

			position: Point{p:Vec3::new(-1.0, 0.0, 0.0)}, // "base" position of camera
			dirNormalized: Normal{v: Vec3::new(1.0, 0.0, 0.0)}, // normalized direction

			upNormalized: Vec3::new(0.0, 1.0, 0.0), // up vector
			sideNormalized: Vec3::new(0.0, 0.0, 1.0), // side vector

			resolutionX: 512,
			resolutionY: 512,
		},

    };



    { // normal rendering
        viewport0.rasterize(&scene);
    }


    // render and write out result image

    // (*) render color image
    let image: Map2d<Color32> = renderColorImage(&scene, &viewport0, &viewport1);

    // (*) write image
    writeColorImage(&image, &format!("img{:06}.ppm", 0));
}





// test scene which is an animation of a orbiting close sphere
// useful for checking if the shadow mapping works correctly

// TODO< let smaller sphere orbit very closely >
fn testscene_closelyOrbitingSphere() {
    for frameNumber in 0..300 {

    






        let mut primitiveSpheres: Vec<PrimitiveSphere> = Vec::new();

        primitiveSpheres = Vec::new();


        primitiveSpheres.push(PrimitiveSphere{
        	shading: Shading{
	    
	            colorR: 0.02,
	            colorG: 0.02,
	            colorB: 1.0,
	        },

            id:0,
            pos:Point{p:Vec3{x:0.0,y:0.0,z:0.5}},
            r:0.2,
        });


        primitiveSpheres.push(PrimitiveSphere{
        	shading: Shading{
	    
	            colorR: 0.02,
	            colorG: 0.02,
	            colorB: 1.0,
	        },

            id:1,
            pos:Point{p:Vec3{x:0.0,y:0.0,z:0.5+0.2*2.0}},
            r:0.2,
        });

        
        primitiveSpheres.push(PrimitiveSphere{
        	shading: Shading{
	            colorR: 0.9,
	            colorG: 0.2,
	            colorB: 0.2,
	        },

            id:2,

            pos:Point{p:Vec3{x:-0.0,y:((frameNumber as f64) * 0.025).sin() * 0.7,z:((frameNumber as f64) * 0.025).cos() * 0.7}},
            r:0.1,
        });


        let mut scene = Scene::new();
        scene.spherePrimitives = primitiveSpheres;





        // viewpor for shadow mapping
        let mut viewport1: Viewport = Viewport{
            rasterized: vec![None; 512*512],

            zBuffer: Map2d::<f64>::new(512, 512, std::f64::INFINITY),

            enableShading: false, // don't enable shading because we are only interested in the depth channel
            
            //  we use back-face culling as a trick against Shadow acne
            //  see http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-16-shadow-mapping/
            face: EnumFace::BACK,




            // the sun is shining in the z direction
            // this is the case because it simplifies the code - we don't need a projection for testing for now



            camera: Camera {
            	type_: EnumCameraType::ORTHOGONAL,

				position: Point{p:Vec3{x:0.0, y:0.0, z:0.0}}, // "base" position of camera
				dirNormalized: Normal{v: Vec3{x:0.0, y:0.0, z:1.0}}, // normalized direction

				upNormalized: Vec3{x:0.0, y:1.0, z:0.0}, // up vector
				sideNormalized: Vec3{x:1.0, y:0.0, z:0.0}, // side vector

				resolutionX: 512,
				resolutionY: 512,
    		},

        };


        { // shadow mapping pass
            viewport1.rasterize(&scene);

            viewport1.processRaymarchingRays();
        }




        








        
        let mut viewport0: Viewport = Viewport{
            rasterized: vec![None; 512*512],

            zBuffer: Map2d::<f64>::new(512, 512, std::f64::INFINITY),

            // we need to render the usual front face for usual visualization
            face: EnumFace::FRONT,

            enableShading: true,


            camera: Camera {
				type_: EnumCameraType::ORTHOGONAL,

				position: Point{p:Vec3::new(-1.0, 0.0, 0.0)}, // "base" position of camera
				dirNormalized: Normal{v: Vec3::new(1.0, 0.0, 0.0)}, // normalized direction

				upNormalized: Vec3::new(0.0, 1.0, 0.0), // up vector
				sideNormalized: Vec3::new(0.0, 0.0, 1.0), // side vector

				resolutionX: 512,
				resolutionY: 512,
    		},

        };



        { // normal rendering
            viewport0.rasterize(&scene);
        }



        if false { // may be used to skip output of image
        	continue;
        }

        // render and write out result image

        // (*) render color image
        let image: Map2d<Color32> = renderColorImage(&scene, &viewport0, &viewport1);

        // (*) write image
        writeColorImage(&image, &format!("img{:06}.ppm", frameNumber));
    }
}






use nalgebra::Perspective3;
use nalgebra::geometry::Reflection;

pub fn main() {
    //testscene_closelyOrbitingSphere();
    //testscene_overlappingSpheres();

    // test perspective projection and rendering
    //testscene_perspectiveSimple1();
    //testscene_perspectiveSimple2();

    // test raytracing
    testscene_raytracingPlane0();

    test_buildAndSerializeBvh();




    




    let mut graphicsEngine = opengl::makeGraphicsEngine().unwrap();
    
    graphicsEngine.initAndAlloc();
    
    let mut t: f64 = -1000000000.0;

    'main: loop {
        for event in graphicsEngine.eventPump.poll_iter() {
            match event {
                sdl2::event::Event::Quit {..} => break 'main,
                _ => {},
            }
        }



        let mut bvhNodes: Vec<opengl::BvhNode> = Vec::new();

        let mut bvhLeafNodes: Vec<opengl::BvhLeafNode> = Vec::new();

        let mut materials: Vec<opengl::Material> = Vec::new();


        // push mesh
        {
            let mut meshVertices = Vec::new();

            let NORM3 = 0.57735026919;

            meshVertices.push(Vec3::new(0.0, 0.0, -NORM3));
            meshVertices.push(Vec3::new(NORM3, 0.0, NORM3));
            meshVertices.push(Vec3::new(-NORM3, NORM3, NORM3));
            meshVertices.push(Vec3::new(-NORM3, -NORM3, NORM3));

            // TODO< store meshTraingleVertexIndices >
            let mut meshVertexIndicesOfPolygons: Vec<[i64; 3]> = Vec::new();
            meshVertexIndicesOfPolygons.push([1, 2, 3]); // bottom

            // sides
            meshVertexIndicesOfPolygons.push([1, 0, 2]);
            meshVertexIndicesOfPolygons.push([2, 0, 3]);
            meshVertexIndicesOfPolygons.push([3, 0, 1]);


            let mut transformationMatrix = Matrix44::new_nonuniform_scaling(&Vector3::new(2.0, 2.0, 2.0));
            transformationMatrix = transformationMatrix * Matrix44::from_euler_angles(0.0, (t as f64) * 0.3 , 0.0);


            for iMeshVertexIndicesOfPoly in meshVertexIndicesOfPolygons {
                let vertexIdx0 = iMeshVertexIndicesOfPoly[0];
                let vertexIdx1 = iMeshVertexIndicesOfPoly[1];
                let vertexIdx2 = iMeshVertexIndicesOfPoly[2];

                let mut vertex0 = meshVertices[vertexIdx0 as usize].clone();
                let mut vertex1 = meshVertices[vertexIdx1 as usize].clone();
                let mut vertex2 = meshVertices[vertexIdx2 as usize].clone();

                // transform vertices
                vertex0 = mul(&transformationMatrix, &vertex0);
                vertex1 = mul(&transformationMatrix, &vertex1);
                vertex2 = mul(&transformationMatrix, &vertex2);

                bvhLeafNodes.push(opengl::BvhLeafNode {
                    nodeType: 1, // 1 is polygon
                    materialIdx: 0,

                    vertex0: Vector4::<f64>::new(vertex0.x, vertex0.y, vertex0.z, 1.0),
                    vertex1: Vector4::<f64>::new(vertex1.x, vertex1.y, vertex1.z, 1.0),
                    vertex2: Vector4::<f64>::new(vertex2.x, vertex2.y, vertex2.z, 1.0),
                });
            }

        }




        // add BVH leaf for testing
        bvhLeafNodes.push(opengl::BvhLeafNode {
            nodeType: 2, // 2 is capped cone
            materialIdx: 0,

            vertex0: Vector4::<f64>::new(2.0, 0.0, 10.0, 1.5), // position and radius
            vertex1: Vector4::<f64>::new(2.01, 3.01, 10.01, 1.503), // position and radius - must be uneven because of precision issues on GPU
            vertex2: Vector4::<f64>::new(0.0, 0.0, 0.0, 0.0),
        });


        
        bvhLeafNodes.push(opengl::BvhLeafNode {
            nodeType: 0, // 0 is sphere
            materialIdx: 0,

            vertex0: Vector4::<f64>::new(2.0, 0.0, 10.0, 1.5), // position and radius
            vertex1: Vector4::<f64>::new(0.0, 0.0, 0.0, 0.0),
            vertex2: Vector4::<f64>::new(0.0, 0.0, 0.0, 0.0),
        });

        bvhLeafNodes.push(opengl::BvhLeafNode {
            nodeType: 0, // 0 is sphere
            materialIdx: 0,

            vertex0: Vector4::<f64>::new(2.0, 3.0, 10.0, 1.5), // position and radius
            vertex1: Vector4::<f64>::new(0.0, 0.0, 0.0, 0.0),
            vertex2: Vector4::<f64>::new(0.0, 0.0, 0.0, 0.0),
        });

        // back plane
        /* commented because we don't need backplane
        bvhLeafNodes.push(opengl::BvhLeafNode {
            nodeType: 0, // 0 is sphere
            materialIdx: 0,

            vertex0: Vector4::<f64>::new(0.0, 0.0, 16.0+1000.0, 1000.0), // position and radius
            vertex1: Vector4::<f64>::new(0.0, 0.0, 0.0, 0.0),
            vertex2: Vector4::<f64>::new(0.0, 0.0, 0.0, 0.0),
        });
        */

        // implicit surface
        bvhLeafNodes.push(opengl::BvhLeafNode {
            nodeType: 3, // 3 is implicit surface
            materialIdx: 0,

            vertex0: Vector4::<f64>::new(0.0, 0.0, 0.0, 0.0), // not used because we are testing
            vertex1: Vector4::<f64>::new(0.0, 0.0, 0.0, 0.0),
            vertex2: Vector4::<f64>::new(0.0, 0.0, 0.0, 0.0),
        });
        
        for i in 0..10 {
            let phase = t * 0.3 + (i as f64) * 0.7;

            let mut positionDelta = Vec3::new(phase.cos(), 0.0, phase.sin());
            positionDelta = positionDelta.scale(3.0);
            let mut position = &Vec3::new(2.0, 0.0, 10.0) + &positionDelta;

            bvhLeafNodes.push(opengl::BvhLeafNode {
                nodeType: 0, // 0 is sphere
                materialIdx: 1,

                vertex0: Vector4::<f64>::new(position.x, position.y, position.z, 0.7), // position and radius
                vertex1: Vector4::<f64>::new(0.0, 0.0, 0.0, 0.0),
                vertex2: Vector4::<f64>::new(0.0, 0.0, 0.0, 0.0),
            });
        }
 

        let mut bvhRootNodeIdx = 0;


        // add material(s)

        materials.push(opengl::Material{
            type_: 0, // lambertian
            fresnelReflectance: 0.1,
            baseColor: Vector3::<f32>::new(1.0, 0.01, 0.01),
        });

        materials.push(opengl::Material{
            type_: 0, // lambertian
            fresnelReflectance: 0.1,
            baseColor: Vector3::<f32>::new(0.1, 0.1, 0.1),
        });

        let camera: opengl::Camera;
        {
            let phase = t * 0.2;

            camera = opengl::Camera{
                position: Vector3::<f32>::new((phase.cos() * 10.0) as f32, 3.0f32, (phase.sin() * 10.0) as f32),
                dir: Vector3::<f32>::new(0.0f32, 0.0f32, 1.0f32), // must be normalized
                up: Vector3::<f32>::new(0.0f32, 1.0f32, 0.0f32) // must be normalized
            }
        }



        graphicsEngine.frame(&bvhNodes, bvhRootNodeIdx, &bvhLeafNodes, &materials, &camera);

        t += (1.0 / 60.0);
    }
}

#[derive(PartialEq)]
enum EnumCameraType {
	ORTHOGONAL,
	PERSPECTIVE,
}

// orthogonal projection "camera"
struct Camera {
	type_: EnumCameraType,

	position: Point, // "base" position of camera
	dirNormalized: Normal, // normalized direction

	upNormalized: Vec3, // up vector
	sideNormalized: Vec3, // side vector

	resolutionX: i64,
	resolutionY: i64,
}

impl Camera {
	// computes the origin and direction for a ray from the (orthogonal) camera
    fn calcRayOriginAndDirection(&self, pixelX: i64, pixelY: i64) -> (Point, Normal) {
    	let sideScale01 = (pixelX as f64) / (self.resolutionX as f64);
    	let upScale01 = (pixelY as f64) / (self.resolutionY as f64);

    	let sideScalem11 = sideScale01 * 2.0 - 1.0;
    	let upScalem11 = upScale01 * 2.0 - 1.0;

    	match self.type_ {
    		EnumCameraType::ORTHOGONAL => {
    			// we just need to compute the parallel direction of the plane

		    	let mut positionP = self.position.p.clone();
		    	positionP = &positionP + &self.sideNormalized.scale(sideScalem11);
		    	positionP = &positionP + &self.upNormalized.scale(upScalem11);

		    	let dirNormalized = self.dirNormalized.clone();

		    	(Point{p:positionP}, dirNormalized)
    		}

    		EnumCameraType::PERSPECTIVE => {
    			// we need to compute the direction of the pixel through the view-plane

    			// how much do we scale the side vector to compute the x vector
    			let scaleSide = 1.0; // HACK< TODO< compute from fov with tangens >

    			// how much do we scale the side vector to compute the y vector
    			let scaleUp = 1.0;  // HACK< TODO< compute from fov with tangens >

    			let mut rayDirection = self.dirNormalized.v.clone();
    			rayDirection = &rayDirection + &self.sideNormalized.scale(sideScalem11 * scaleSide);
    			rayDirection = &rayDirection + &self.upNormalized.scale(upScalem11 * scaleUp);

    			// we need to normalize
    			let dirNormalized = normalize(&rayDirection);

    			(self.position.clone(), Normal{v:dirNormalized})
    		}
    	}

    }


	// projects the position into the space of the camera
	// note that z - depth is the world depth, _NOT_ the depth of a rasterized camera!
	fn project(&self, position: &Vec3) -> Vec3 {
		// just implemented for orthogonal projection because we need it for non-perspective shadow mapping
	    assert!(self.type_ == EnumCameraType::ORTHOGONAL, "only implemented for orthogonal projection!");

	    let diff = Vec3{x:position.x-self.position.p.x, y:position.y-self.position.p.y, z:position.z-self.position.p.z};

	    let projectedX = dot(&diff, &self.sideNormalized);
	    let projectedY = dot(&diff, &self.upNormalized);
	    let projectedZ = dot(&diff, &self.dirNormalized.v);

	    Vec3{x:projectedX, y:projectedY, z:projectedZ}
	}


	// method to compute the depth of a projected point
	fn calcDepthOfProjectedPoint(&self, p: &Point) -> f64 {
		match self.type_ {
    		EnumCameraType::ORTHOGONAL => {
    			let positionDiff = &p.p - &self.position.p; // direction
    			return dot(&self.dirNormalized.v, &positionDiff); // distance by definition of orthogonal projection
    		}

    		EnumCameraType::PERSPECTIVE => {
    			let mut worldPosition = p.p.clone();
				
				// we need to compute the z(depth value)
				//  we can do it with just the dot product because we just care about z
				let diff = &worldPosition - &self.position.p;
				return dot(&self.dirNormalized.v, &diff);
    		}
    	}
	}

	// direction of view, is the vector from the collision point to the camera "eye"
	fn retViewDirOfPixel(&self, pixelX: i64, pixelY: i64) -> Normal {
		match self.type_ {
    		EnumCameraType::ORTHOGONAL => {
    			return Normal{v:self.dirNormalized.v.scale(-1.0)};
    		}

    		EnumCameraType::PERSPECTIVE => {
    			let sideScale01 = (pixelX as f64) / (self.resolutionX as f64);
    			let upScale01 = (pixelY as f64) / (self.resolutionY as f64);

    			let sideScalem11 = sideScale01 * 2.0 - 1.0;
    			let upScalem11 = upScale01 * 2.0 - 1.0;



    			// how much do we scale the side vector to compute the x vector
    			let scaleSide = 1.0; // HACK< TODO< compute from fov with tangens >

    			// how much do we scale the side vector to compute the y vector
    			let scaleUp = 1.0;  // HACK< TODO< compute from fov with tangens >

    			let mut rayDirection = self.dirNormalized.v.clone();
    			rayDirection = &rayDirection + &self.sideNormalized.scale(sideScalem11 * scaleSide);
    			rayDirection = &rayDirection + &self.upNormalized.scale(upScalem11 * scaleUp);

    			// we need to normalize
    			let dirNormalized = normalize(&rayDirection);


    			return Normal{v:dirNormalized.scale(-1.0)};
    		}
    	}
	}
}

// testing function - is hardcoded for a specific scene under specific assumptions
// raymarching/raytracing entry for shadow ray

// /param pixelX x index of the pixel
// /param pixelY y index of the pixel
// /result t(depth) of ray or None if it didn't hit anything
fn rayEntry_ShadowRay_testing(orthogonalCamera: &Camera, pixelX: i64, pixelY: i64) -> Option<f64> {
	// bilinear patch for testing
	let testPatch: Bilinear = Bilinear{
		// TESTING< for now we set it to the same values so it acts like a plane >

		_0: Linear{a:0.1, b:0.1},
    	_1: Linear{a:0.1, b:0.1}
	};

	let patchRaymarchingSteps = 500;

	// compute the start position and normalized direction of the ray from the camera
	let (pStart, dirNormalized) = orthogonalCamera.calcRayOriginAndDirection(pixelX, pixelY);

	// do the raymarching for the implicit surface
	// call into the domain mapping function because x and y are in range [-1.0;1.0] and have to get mapped to [0.0;1.0]
	let raymarchingPatchRayResult:Option<(f64, Normal)> = raymarchPatchDomainM11(
    	&pStart, // pStart
    	&dirNormalized.v, // dir
    	patchRaymarchingSteps, // steps
    	&testPatch
	);

	if raymarchingPatchRayResult.is_some() {
		// we just need to return the depth
		let (rayDepth, normal) = raymarchingPatchRayResult.unwrap();
		return Some(rayDepth)
	}
	None
}














fn linear(t: f64, a: f64, b:f64) -> f64 {
    let diff = b - a;
    a + diff * t
}

// bilinear interpolation
// d is depth and indices are y_x
fn bilinear(t: Vec2, d00: f64, d01: f64, d10: f64, d11: f64) -> f64 {
    let d0 = linear(t.x, d00, d01);
    let d1 = linear(t.x, d10, d11);
    
    let res = linear(t.y, d0, d1);

    res
}


pub struct Vec2 {
    x: f64,
    y: f64,
}


impl Vec2 {
	pub fn new(x: f64, y: f64) -> Vec2 {
		Vec2{x:x,y:y}
    }

    pub fn scale(&self, s: f64) -> Vec2 {
    	Vec2{x:self.x*s,y:self.y*s}
    }

    pub fn magnitudeSquared(&self) -> f64 {
    	self.x*self.x + self.y*self.y
    }

    pub fn magnitude(&self) -> f64 {
    	self.magnitudeSquared().sqrt()
    }

    pub fn normalized(&self) -> Vec2 {
    	let m = self.magnitude();
    	self.scale(1.0/m)
    }
}

impl<'a, 'b> std::ops::Add<&'b Vec2> for &'a Vec2 {
    type Output = Vec2;

    fn add(self, rhs: &'b Vec2) -> Vec2 {
        Vec2{x:self.x + rhs.x, y:self.y + rhs.y}
    }
}

impl<'a, 'b> std::ops::Sub<&'b Vec2> for &'a Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: &'b Vec2) -> Vec2 {
        Vec2{x:self.x - rhs.x, y:self.y - rhs.y}
    }
}

fn dot2d(a: &Vec2, b: &Vec2) -> f64 {
	a.x*b.x + a.y*b.y
}


// structure for linear interpolated segments
pub struct Linear {
    a: f64,
    b: f64,
}

pub struct Bilinear {
    _0: Linear,
    _1: Linear,
}


// compute the depth of a (normalized) point in bilinear space
fn calcDepthBilinear(p: &Vec3, b: &Bilinear) -> f64 {
    bilinear(Vec2{x:p.x, y:p.y}, b._0.a, b._0.b, b._1.a, b._1.b)
}

// computes the normal of a (normalized) point in bilinear space
fn calcNormalBilinear(p: &Vec3, b: &Bilinear) -> Normal {
	// helper function for computing the derivative of a linear segment
	fn calcDerivativeY(l: &Linear) -> f64 {
		l.b - l.a
	}

    // compute derivates
    let change_x0:f64 = calcDerivativeY(&b._0);// the change of a line is simply the difference
    let change_x1:f64 = calcDerivativeY(&b._1);

    // linear interpolate
    let change_x = linear(p.y, change_x0, change_x1);



    let posY_0 = linear(p.x, b._0.a, b._0.b);
    let posY_1 = linear(p.x, b._1.a, b._1.b);
    
    // we can just compute the difference because the domain is [0.0; 1.0]
    let change_y = calcDerivativeY(&Linear{a:posY_0, b:posY_1});

    // negations because the normal has to point away from the surface
	Normal{v:Vec3::new(-change_x, -change_y, 1.0)}
}



// entry which maps x and y from a range [-1.0;1.0] to [0.0;1.0]
fn raymarchPatchDomainM11(
    pStart: &Point,
    dir: &Vec3,
    steps: i64,
    patch: &Bilinear
) -> Option<(f64, Normal)> {

	// map domain
	let x = (pStart.p.x + 1.0) * 0.5;
	let y = (pStart.p.y + 1.0) * 0.5;
	let pStartMapped = Point{p:Vec3{x:x, y:y, z:pStart.p.z}};

	raymarchPatch(
		&pStartMapped,
		dir,
    	steps,
    	patch
	)
}



// we have the ability to compute intersections with analytical surfaces like
// * bilinear patches
// * TODO< more complicated patches >

// TODO GPU< implement on GPU with texture lookup >

// these computations can be accelerated on the GPU with a texture lookup
// see https://blog.demofox.org/2016/02/22/gpu-texture-sampler-bezier-curve-evaluation/
//     https://blog.demofox.org/2016/12/16/analyticsurfacesvolumesgpu/
//     http://demofox.org/TextureSamplerSurface.html

// /param pStart start position in the patch
//        x range is [0.0; 1.0]
//        y range is [0.0; 1.0]
//        z range depends on th patch and can be anything
// TODO< make to use generic trait >

// /result ray-"time" and the normal
fn raymarchPatch(
    pStart: &Point,
    dir: &Vec3,
    steps: i64,
    patch: &Bilinear
) -> Option<(f64, Normal)> {
	// TODO< optimize by using variable step size >


    // checkBoundsIteration check the bounds while iterating - mut be true if we didn't constraint the iteration depth
    let checkBoundsIteration = true; // enable by default because 

    let mut magnitudeOfStepsize = 0.03;

    let step = &dir.scale(magnitudeOfStepsize);
    
    let mut p = pStart.p.clone();

    let mut t = 0.0;

    // we use the sign to check if we intersected with the surface
    let signEntry;
    {
        let depthOfSurface = calcDepthBilinear(&p, patch);

        signEntry = (p.z - depthOfSurface).signum();
    }

    for _step in 0..steps {
        if checkBoundsIteration {
            if !inRange01(p.x) || !inRange01(p.y) {
                // it is not valid to compute the patch outside of the range
                // so we just extend the ray

                p = &p + &step;
                t += magnitudeOfStepsize;
                continue;
            }
        }

        let depthOfSurface = calcDepthBilinear(&p, patch);

        let sign = (p.z - depthOfSurface).signum();

        if sign != signEntry {
            // we intersected if the sign is different

            // we need to compute the normal
            let normal = calcNormalBilinear(&p, patch);

            return Some((t, normal));
        }

        p = &p + &step;
        t += magnitudeOfStepsize;
    }

    // no intersection
    None
}


// check if number is between 0.0 and 1.0 inclusive
fn inRange01(v: f64) -> bool {
    return (v - 0.5).abs() <= 0.5
}







use nalgebra::{U4, Matrix, MatrixArray, Vector4, Vector3};

// type for a 4x4 matrix
type Matrix44 = Matrix<f64, U4, U4, MatrixArray<f64, U4, U4>>;

type Vec4 = Vector4<f64>;

// matrix with inversion
// used for fast tranformation in either direction without the difficult inversion
struct DualMatrix44 {
	m: Matrix44,
	inv: Matrix44
}

fn createTranslationMatrix44(x: f64, y: f64, z: f64) -> Matrix44 {
	Matrix44::new_translation(&Vector3::new(x, y, z))
}

fn createDualTranslationMatrix44(x: f64, y: f64, z: f64) -> DualMatrix44 {
	let m = Matrix44::new_translation(&Vector3::new(x, y, z));
	let inv = Matrix44::new_translation(&Vector3::new(-x, -y, -z));
	DualMatrix44{m:m, inv:inv}
}

//fn createScalingMatrix44(x: f64, y: f64, z: f64) -> Matrix44 {
//	Matrix44::new_scaling(Vector3::new(x, y, z))
//}


fn mul(m: &Matrix44, vector: &Vec3) -> Vec3 {
	let vec4 = Vector4::new(vector.x, vector.y, vector.z, 1.0);
	let resultVec4 = m * vec4;
	// TODO MAYBE< use .xyz() when we use everywhere Vector3 as Vec3 >
	Vec3::new(resultVec4.x, resultVec4.y, resultVec4.z)
}

// point structure to identify a point in space
#[derive(Clone)] // cloning is cheap
pub struct Point {
	p: Vec3
}

impl Point {
	fn new(x: f64, y: f64, z: f64) -> Point {
		Point{p:Vec3::new(x,y,z)}
	}
}

// normal structure to identify a normal in space
#[derive(Clone)] // cloning is cheap
pub struct Normal {
	pub v: Vec3 // v stands for vector
}

// TODO< transform point by inverse of transformation matrix and let the ray collide with the bilinear surface >




pub struct Map2d<T> {
	arr: Vec<T>,

	width: i64,
}

impl <T: Clone> Map2d<T> {
	pub fn new(height: i64, width: i64, value: T) -> Map2d<T> {
		// TODO< panic when width or height are negative! >

		let arr = vec![value; (width*height) as usize];

        Map2d::<T>{arr: arr, width: width}
    }

	// computes the origin and direction for a ray from the (orthogonal) camera
    pub fn retAtUnchecked(&self, y: i64, x: i64) -> &T {
    	&self.arr[(y * self.width + x) as usize]
    }

    pub fn setAtUnchecked(&mut self, y: i64, x: i64, value: &T) {
    	self.arr[(y * self.width + x) as usize] = (*value).clone();
    }

    pub fn isInBounds(&self, y: i64, x: i64) -> bool {
    	true &&
    		y >= 0 && y < self.retHeight() &&
    		x >= 0 && x < self.retWidth()
    }

    pub fn retWidth(&self) -> i64 {
    	self.width
    }

    pub fn retHeight(&self) -> i64 {
    	self.arr.len() as i64 / self.width 
    }
}




struct Plane {
	n: Normal, // normal
	center: Point,
}

fn calcRayPlane(rayOrigin: &Point, rayDir: &Normal, plane: &Plane) -> Option<f64> {
	// from https://stackoverflow.com/a/23976134/388614

	// TODO< optimize by precalculating d >
	let denom = dot(&plane.n.v, &rayDir.v);
	if denom.abs() > /*epsilon*/0.0001 {
    	let t = dot(&(&plane.center.p - &rayOrigin.p), &plane.n.v) / denom;
    	return Some(t)
	}
	None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planeEquation() {
        let testPlane = Plane {
			n: Normal{v:Vec3::new(1.0, 0.0, 0.0)},
			center: Point{p:Vec3::new(0.5, 0.0, 0.0)}
		};

		let intersectionOption = calcRayPlane(&Point{p:Vec3::new(-1.0, 0.0, 0.0)}, &Normal{v:Vec3::new(1.0, 0.0, 0.0)}, &testPlane);

		assert!(intersectionOption.is_some()); // must be a valid intersection
		assert!(intersectionOption.unwrap() == 1.5);
    }



    #[test]
    fn test_QuadPlane() {
		// construction of quad plane by 3 points (which must be perpendicular)
		let basePoint = Point{p:Vec3::new(1.0, 0.0, 0.0)};
		let a = Point{p:Vec3::new(2.0, 0.0, 0.0)};
		let b = Point{p:Vec3::new(1.0, 1.0, 0.0)};
		
		let testPlane:QuadPlane = makeQuadPlaneFromPoints(&basePoint, &a, &b);

		assert!(calcRayQuadPlane(&Point{p:Vec3::new(0.0,0.0, -1.0)}, &Normal{v:Vec3::new(0.0, 0.0, 1.0)}, &testPlane).is_none());

		assert!(calcRayQuadPlane(&Point{p:Vec3::new(3.0,0.0, -1.0)}, &Normal{v:Vec3::new(0.0, 0.0, 1.0)}, &testPlane).is_none());

		// center
		assert!(calcRayQuadPlane(&Point{p:Vec3::new(1.5,0.5, -1.0)}, &Normal{v:Vec3::new(0.0, 0.0, 1.0)}, &testPlane).is_some());


		// edge hit test
		assert!(calcRayQuadPlane(&Point{p:Vec3::new(1.0,0.0, -1.0)}, &Normal{v:Vec3::new(0.0, 0.0, 1.0)}, &testPlane).is_some());

		// edge hit test
		assert!(calcRayQuadPlane(&Point{p:Vec3::new(2.0,0.0, -1.0)}, &Normal{v:Vec3::new(0.0, 0.0, 1.0)}, &testPlane).is_some());

		// edge hit test
		assert!(calcRayQuadPlane(&Point{p:Vec3::new(2.0,1.0, -1.0)}, &Normal{v:Vec3::new(0.0, 0.0, 1.0)}, &testPlane).is_some());

		// edge hit test
		assert!(calcRayQuadPlane(&Point{p:Vec3::new(1.0,1.0, -1.0)}, &Normal{v:Vec3::new(0.0, 0.0, 1.0)}, &testPlane).is_some());
	}


	// TODO< test bounding box intersection >
}


#[derive(Clone)]
struct Box {
	extend: Vec3,
}

struct BoundingBox {
	box_: Box, // a bounding box is a box with a position and orientation

	position: Point,

	up: Normal,
	front: Normal,

	faces: Option<[QuadPlane; 6]>,
}

impl BoundingBox {
	fn calcSideNormalized(&self) -> Normal {
		Normal{v:normalize(&cross(&self.up.v, &self.front.v))}
	}
}

// returns a point on the unit box
fn calcUnitBoxPoint(xi: i64, yi:i64, zi:i64) -> Vec3 {
	debug_assert!(xi.abs()<=1);
	debug_assert!(yi.abs()<=1);
	debug_assert!(zi.abs()<=1);

	Vec3::new(xi as f64, yi as f64, zi as f64)
}

// return the point on the surface of a cube defined by it's three (orthogonal axis) offsets
/*
fn calcBoxPoint(box_: &Box, xi: i64, yi:i64, zi:i64) -> Vec3 {
	debug_assert!(xi.abs()<=1);
	debug_assert!(yi.abs()<=1);
	debug_assert!(zi.abs()<=1);

	Vec3::new(box_.extend.x * (xi as f64), box_.extend.y * (yi as f64), box_.extend.z * (zi as f64))
}*/

// compute world space point of bounding box
/* commented because it is wrong and doesn't take the orientation into account
fn calcBoundingBoxPoint(boundingBox: &BoundingBox, xi: i64, yi:i64, zi:i64) -> Point {
	let mut rel = calcBoxPoint(&boundingBox.box_, xi, yi, zi);
	rel = Vec3::new(rel.x * boundingBox.box_.extend.x, rel.y * boundingBox.box_.extend.y, rel.z * boundingBox.box_.extend.z);
	Point{p:&boundingBox.position.p + &rel}
}
*/

// computes closest point of ray vs bounding box
fn calcRayBoundingBox(rayOrigin: &Point, rayDir: &Normal, boundingBox: &BoundingBox) -> Option<f64> {
	match &boundingBox.faces {
		Some(faces) => {
			let mut res: Option<f64> = None;

			// TODO< map collision result >
			let rayIntersectionResults = vec![
				calcRayQuadPlane(&rayOrigin, &rayDir, &faces[0]),
				calcRayQuadPlane(&rayOrigin, &rayDir, &faces[1]),
				calcRayQuadPlane(&rayOrigin, &rayDir, &faces[2]),
				calcRayQuadPlane(&rayOrigin, &rayDir, &faces[3]),
				calcRayQuadPlane(&rayOrigin, &rayDir, &faces[4]),
				calcRayQuadPlane(&rayOrigin, &rayDir, &faces[5])
			];

			for iBoundingBox in rayIntersectionResults {
				if iBoundingBox.is_none() {
					continue;
				}

				if res.is_none() {
					if iBoundingBox.unwrap() >= 0.0 { // must be in front
						// replace
						res = iBoundingBox;
					}
				}

				debug_assert!(iBoundingBox.is_some() && res.is_some());

				// check cloest one
				let depthOrRes = res.unwrap();
				let depthOfIBoundingBox = iBoundingBox.unwrap();

				if depthOfIBoundingBox < depthOrRes && depthOfIBoundingBox >= 0.0 { // must be in front and closer
					res = iBoundingBox;
				}
			}

			res
		}
		None => {
			// no faces -> thus no intersection
			// ASK< should this invalid and lead to a debugging assertion? >
			return None;
		}
	}
}



// is a rectangular patch of a plane
struct QuadPlane {
	plane: Plane,

	basePoint: Point,
	tangentNormalized: Normal,
	cotangentNormalized: Normal,

	extendTangent: f64, // extend on tangent
	extendCotangent: f64 // extend on cotangent
}

// construction of quad plane by 3 points (which must be perpendicular)
fn makeQuadPlaneFromPoints(base: &Point, a: &Point, b: &Point) -> QuadPlane {
	let aBase = &a.p - &base.p;
	let bBase = &b.p - &base.p;

	let aBaseNormalized = normalize(&aBase);
	let bBaseNormalized = normalize(&bBase);
	
	let normal = cross(&aBaseNormalized, &bBaseNormalized);

	// TODO< check that aBaseNormalized and bBaseNormalized are perpendicular >

	QuadPlane {
		plane: Plane { // plane is aligned in the normal and has the base point as the center
			n: Normal{v:normal},
			center: base.clone()
		},

		basePoint: base.clone(),
		tangentNormalized: Normal{v:aBaseNormalized.clone()},
		cotangentNormalized: Normal{v:bBaseNormalized.clone()},

		extendTangent: aBase.magnitude(),
		extendCotangent: bBase.magnitude(),
	}
}


// intersection test of QuadPlane vs ray
fn calcRayQuadPlane(rayOrigin: &Point, rayDir: &Normal, quadPlane: &QuadPlane) -> Option<f64> {
	let planeIntersectionOption: Option<f64> = calcRayPlane(&rayOrigin, &rayDir, &quadPlane.plane);
	if !planeIntersectionOption.is_some() {
		return None;
	}

	let p = &rayOrigin.p + &rayDir.v.scale(planeIntersectionOption.unwrap());

	// (*) project position on tangent and cotangent
	
	let pBase = &p - &quadPlane.basePoint.p;

	let projectionTangent = dot(&pBase, &quadPlane.tangentNormalized.v);
	let projectionCotangent = dot(&pBase, &quadPlane.cotangentNormalized.v);

	// (*) check range inclusive
	if projectionTangent < 0.0 || projectionTangent > quadPlane.extendTangent {
		return None;
	}

	if projectionCotangent < 0.0 || projectionCotangent > quadPlane.extendCotangent {
		return None;
	}

	planeIntersectionOption
}




// TODO< compute planes of bounding box >




// recomputes the AABB of the bounding box after transforming it 
fn recalcAabb(bb: &BoundingBox, m: &Matrix44) -> BoundingBox {
	/*
	 steps to compute the AABB for the implicit surface

	 (*) we need to compute the AABB wwhich encloses the implicit surface
	     compute vertex locations of AABB with real size at origin

	 (*) transform points after global transformation matrix
	 */

	// comented because unnecessary
	//let mut bb2:BoundingBox = transformBoundingBox(&bb, &m);

	// (*) recompute axis and extend by spanning up the AABB from the new points

	let mut points: Vec<Point> = Vec::new();
	for zSign in vec![-1,1] {
		for ySign in vec![-1,1] {
			for xSign in vec![-1,1] {
        		let pointOnUnitBox = calcUnitBoxPoint(xSign, ySign, zSign);
        		// scale
        		// BUG BUG BUG< extend needs to be multiplied by half >
        		let pointOnBoundingBox = Vec3::new(pointOnUnitBox.x * bb.box_.extend.x, pointOnUnitBox.y * bb.box_.extend.y, pointOnUnitBox.z * bb.box_.extend.z);

        		// transform
        		let transformedPoint = mul(&m, &pointOnBoundingBox);

        		points.push(Point{p:transformedPoint});
    		}
    	}
    }

    // compute new AABB of points in absolute coordinates
    // + computee absolute aabb bounds in world-space
    let mut aabbMin: Vec3 = Vec3::new(std::f64::INFINITY, std::f64::INFINITY, std::f64::INFINITY);
    let mut aabbMax: Vec3 = Vec3::new(-std::f64::INFINITY, -std::f64::INFINITY, -std::f64::INFINITY);

    for iPoint in points.iter() {
    	aabbMin = Vec3::new(aabbMin.x.min(iPoint.p.x), aabbMin.y.min(iPoint.p.y), aabbMin.z.min(iPoint.p.z));
    	aabbMax = Vec3::new(aabbMax.x.max(iPoint.p.x), aabbMax.y.max(iPoint.p.y), aabbMax.z.max(iPoint.p.z));
    }

    // + compute bounding
    debug_assert!(aabbMin.x <= aabbMax.x);
    debug_assert!(aabbMin.y <= aabbMax.y);
    debug_assert!(aabbMin.z <= aabbMax.z);

    let aabbExtend = &aabbMax - &aabbMin;

	BoundingBox {
		box_: Box{
			extend: aabbExtend.clone()
		},

		position: Point{p:(&aabbMax + &aabbMin).scale(0.5)},

		up: Normal{v:Vec3::new(0.0, 1.0, 0.0)},
		front: Normal{v:Vec3::new(0.0, 0.0, 1.0)},

		faces: None,
	}
}

/* commented because not needed

fn transformBoundingBox(bb: &BoundingBox, m: &Matrix44) -> BoundingBox {
	let newPosition = Point{p:mul(&m, &bb.position.p)};

	// TODO< transform normal directly as described in book about photorealistic renderer >
	// small HACK to compute transformed up and front
	// we add position to the up and side vector, transform it, compute the vectors back, normalize
	let mut upTranslated = &bb.position.p + &bb.up.v;
	let mut frontTranslated = &bb.position.p + &bb.front.v;
	
	upTranslated = mul(&m, &upTranslated);
	frontTranslated = mul(&m, &frontTranslated);

	upTranslated = &upTranslated - &newPosition.p;
	frontTranslated = &frontTranslated - &newPosition.p;

	upTranslated = normalize(&upTranslated);
	frontTranslated = normalize(&frontTranslated);


	BoundingBox {
		box_: bb.box_.clone(),
		position: newPosition,
		up: Normal{v:upTranslated},
		front: Normal{v:frontTranslated},
		faces:None, // none means that it need to be recomputed
	}
}
*/







/*
 steps after collision with AABB took place and ray has to get shot 

 * (*) transform global intersection position with inverse matrix to local coordinate system

 * (*) do ray marching

 * (*) after ray marching is done we have to transform the normal and local position back to globals

 */

















struct ProjectionResult {
    area: f64,
    center: Vec2,
    axisA: Vec2,
    axisB: Vec2,	
	a: f64, b: f64, c: f64, d: f64, e: f64, f: f64,
}

// The MIT License
// Copyright © 2014 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Analytic projection of a sphere to screen pixels. 

// Spheres in world space become ellipses when projected to the camera view plane. In fact, these
// ellipses can be analytically determined from the camera parameters and the sphere geometry,
// such that their exact position, orientation and surface area can be compunted. This means that,
// given a sphere and a camera and buffer resolution, there is an analytical formula that 
// provides the amount of pixels covered by a sphere in the image. This can be very useful for
// implementing LOD for objects based on their size in screen (think of trees, vegetation, characters
// or any other such complex object).

// This provides too the center and axes of the ellipse

// example code/source: https://www.shadertoy.com/view/XdBGzd
// More info, here: http://www.iquilezles.org/www/articles/sphereproj/sphereproj.htm

fn projectSphere(
	/* sphere        */ sphere: &Vec4,
	                    cameraMat: &Matrix44, /* camera matrix (world to camera) */
	                    fle: f64 /* projection (focal length) - FOV (probably in units) */
) -> ProjectionResult 
{
    // transform to camera space	
	let o:Vec3 = mul(&cameraMat, &Vec3::new(sphere.x,sphere.y,sphere.z));
	
    let r2 = sphere.w*sphere.w;
	let z2 = o.z*o.z;	
	let l2 = dot(&o, &o);
	
	// axis
	let axa: Vec2 = Vec2::new( o.x,o.y).scale(fle*(-r2*(r2-l2)/((l2-z2)*(r2-z2)*(r2-z2))).sqrt());
	let axb: Vec2 = Vec2::new(-o.y,o.x).scale(fle*(-r2*(r2-l2)/((l2-z2)*(r2-z2)*(r2-l2))).sqrt());

	let area = -3.141593*fle*fle*r2*(((l2-r2)/(r2-z2)).abs()).sqrt()/(r2-z2);

	// alternative formula for area - commented because it is slower
    //area = length(axa)*length(axb)*3.141593;
	
	// center
	let center = o.xy().scale(fle*o.z/(z2-r2));


	ProjectionResult{
		area: area, 
		center: center,

		axisA: axa,
		axisB: axb, 
        
        /* implicit ellipse f(x,y) = aÂ·xÂ² + bÂ·yÂ² + cÂ·xÂ·y + dÂ·x + eÂ·y + f = 0 */
        /* a */ a: r2 - (o.y*o.y + z2),
        /* b */ b: r2 - (o.x*o.x + z2),
        /* c */ c: 2.0*o.x*o.y,
        /* d */ d: 2.0*o.x*o.z*fle,
        /* e */ e: 2.0*o.y*o.z*fle,
        /* f */ f: (r2-l2+z2)*fle*fle
    }
}






// computes the relative distance to the center [0.0; 1.0] of an ellipse
// /param rel relative position relative to center of ellipse
// /param axisA not normalized axis A, perpendicular to B
// /param axisB not normalized axis B, perpendicular to A
fn calcEllipseDistToCenter(rel: &Vec2, axisA: &Vec2, axisB: &Vec2) -> f64 {
	let projectedA = dot2d(&rel, &axisA.normalized()) / axisA.magnitude();
	let projectedB = dot2d(&rel, &axisB.normalized()) / axisB.magnitude();

	return calcDistToCenter(&Vec2::new(projectedA, projectedB))
}

fn calcDistToCenter(rel: &Vec2) -> f64 {
	rel.magnitude()
}

fn project2(base: &Vec2, rel: &Vec2) -> Vec2 {
	let scale_ = dot2d(&rel, &base.normalized()) / base.magnitude();
	base.scale(scale_)
}





////////////////////
// shading mathematics

// reflection of vector
fn reflect(d: &Vec3, n: &Normal) -> Normal {
	// see https://math.stackexchange.com/a/13263/87975
	Normal{v:d - &n.v.scale(-2.0 * dot(d, &n.v))}
}






// more intersections

fn dot2(v: &Vec3) -> f64 {
	dot(&v,&v)
}

// Intersection of a ray and a capped cone oriented in an arbitrary direction
fn iCappedCone(ro: &Vec3, rd: &Vec3, 
               pa: &Vec3, pb: &Vec3, 
               ra: f64, rb: f64) -> Vec4
{

	// The MIT License
	// Copyright © 2016 Inigo Quilez
	// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

	// code https://www.shadertoy.com/view/llcfRf
	// ref https://www.iquilezles.org/www/articles/intersectors/intersectors.htm

	// see the GLSL version at https://www.shadertoy.com/view/llcfRf 

    let ba:Vec3 = pb - pa;
    let oa:Vec3 = ro - pa;
    let ob:Vec3 = ro - pb;
    
    let baba:f64 = dot(&ba,&ba);
    let rdba:f64 = dot(&rd,&ba);
    let oaba:f64 = dot(&oa,&ba);
    let obba:f64 = dot(&ob,&ba);
    
    //caps
    if oaba<0.0 {
        // example of delayed division
        if dot2(&(&oa.scale(rdba)-&rd.scale(oaba)))< ra*ra*rdba*rdba {
        	let normal:Vec3 = ba.scale(-inversesqrt(baba));
        	return Vector4::new(-oaba/rdba, normal.x, normal.y, normal.z);
        }
    }
    else if obba>0.0
    {
        // example of NOT delayed division
    	let t:f64 =-obba/rdba;
        if dot2(&(&ob+&rd.scale(t))) < rb*rb {
        	let normal:Vec3 = ba.scale(inversesqrt(baba));
        	return Vector4::new(t, normal.x, normal.y, normal.z);
        }
    }
    
    
    // body
    let rr:f64 = rb - ra;
    let hy:f64 = baba + rr*rr;
    let oc:Vec3 = &oa.scale(rb) - &ob.scale(ra);
    
    let ocba:f64 = dot(&oc,&ba);
    let ocrd:f64 = dot(&oc,&rd);
    let ococ:f64 = dot(&oc,&oc);
    
    let k2:f64 = baba*baba      - hy*rdba*rdba; // the gap is rdrd which is 1.0
    let k1:f64 = baba*baba*ocrd - hy*rdba*ocba;
    let k0:f64 = baba*baba*ococ - hy*ocba*ocba;

    let h:f64 = k1*k1 - k2*k0;
    if h<0.0 {
    	return Vector4::new(-1.0,-1.0,-1.0,-1.0);
    }

    let t:f64 = (-k1-sign(rr)*h.sqrt())/(k2*rr);
    
    let y:f64 = oaba + rdba*t;
    if y>0.0 && y<baba {
    	let insideNormalize:Vec3 = &((&(&oa+&rd.scale(t)).scale(baba)-&ba.scale(rr*ra)).scale(baba))-&ba.scale(hy*y);
    	let normal:Vec3 = normalize(&insideNormalize);
        return Vector4::new(t, normal.x, normal.y, normal.z);
    }
    
    return Vector4::new(-1.0,-1.0,-1.0,-1.0);
}


// function with GLSL names
fn inversesqrt(v:f64) -> f64 {
	1.0/v.sqrt()
}

fn sign(v:f64) -> f64 {
	if v >= 0.0 {
		1.0
	}
	else {
		-1.0
	}
}






// center of mass is simply 
pub trait HasCenter {
	fn retCenter(&self) -> Point;
}




mod bvh {
	use super::{HasCenter, HasAabb};
	use super::{Vec3, Point};
	use super::calcAabbMerge;

	use std::rc::Rc;


    
    enum EnumAxis {
	    X, Y, Z
	}

	// sort positions by axis
	fn sortByAxis<T>(ns: &mut Vec<Rc<T>>, axis: &EnumAxis) where T: HasCenter {
	    match axis {
	        EnumAxis::X => {ns.sort_by(|a, b| a.retCenter().p.x.partial_cmp(&b.retCenter().p.x).unwrap());}
	        EnumAxis::Y => {ns.sort_by(|a, b| a.retCenter().p.y.partial_cmp(&b.retCenter().p.y).unwrap());}
	        EnumAxis::Z => {ns.sort_by(|a, b| a.retCenter().p.z.partial_cmp(&b.retCenter().p.z).unwrap());}
	    }
	}

    // splits elements into two lists if possible
	fn split<T>(elements: &Vec<Rc<T>>) -> (Vec<Rc<T>>, Vec<Rc<T>>)
		where T: HasCenter
	{
	    assert!(elements.len() > 0);
	    
	    if elements.len() == 1 {
	        return (vec![elements[0].clone()], vec![elements[0].clone()]);
	    }
	    else if elements.len() == 2 {
	        return (vec![elements[0].clone()], vec![elements[1].clone()]);
	    }

	    let midIdx = elements.len() / 2;
	    return (elements[..midIdx].to_vec(), elements[midIdx..].to_vec());
	}



	extern crate rand;
	use rand::{Rng};
	use rand::rngs::ThreadRng;

	pub fn splitAndCreateBvhNode<T>(elements: &mut Vec<Rc<T>>, rng: &mut rand::rngs::ThreadRng) -> Option<Rc<BvhNode<T>>>
		where T: HasCenter+HasAabb
	{
		if elements.len() == 0 {
			None
		}
		else {
			Some(splitAndCreateBvhNodeInternal(elements, rng))
		}
	}

	fn splitAndCreateBvhNodeInternal<T>(elements: &mut Vec<Rc<T>>, rng: &mut rand::rngs::ThreadRng) -> Rc<BvhNode<T>>
		where T: HasCenter+HasAabb
	{
		assert!(elements.len() > 0); // case of 0 is not handled!
		                             // is not an issue because the entry is splitAndCreateBvhNode

		// implemented algorithm is really simple
		// we split at random axis without any metrics

		if elements.len() == 1 {
			return Rc::new(BvhNode::LEAF{
				bvhCenter: elements[0].retAabbCenter(),
				bvhExtend: elements[0].retAabbExtend(),
				element: Rc::clone(&elements[0])
			});
		}

		// we need to split

		let axisIdx: i32 = rng.gen_range(0, 3);
		let axis = match axisIdx {
			0 => {EnumAxis::X}
			1 => {EnumAxis::Y}
			_ => {EnumAxis::Z}
		};

		sortByAxis(elements, &axis);
		let (mut left, mut right) = split(elements);

		// recursion
		let leftBvhNode = splitAndCreateBvhNodeInternal(&mut left, rng);
		let rightBvhNode = splitAndCreateBvhNodeInternal(&mut right, rng);

		let (mergedAabbCenter, mergedAabbExtend) = calcAabbMerge(elements);

		Rc::new(BvhNode::BRANCH{
			bvhCenter: mergedAabbCenter,
			bvhExtend: mergedAabbExtend,
			left: leftBvhNode,
			right: rightBvhNode
		})
	}



	// BVH node
	pub enum BvhNode<T> where T: HasCenter+HasAabb {
		LEAF {bvhCenter: Point, bvhExtend: Vec3, element: Rc<T>},
		BRANCH {bvhCenter: Point, bvhExtend: Vec3,  left: Rc<BvhNode<T>>, right: Rc<BvhNode<T>>}
	}

}

// TODO< we need trait for extend >
pub trait HasAabb {
	fn retAabbExtend(&self) -> Vec3;
	//fn setAabbExtend(&mut self, extend: &Vec3);

	fn retAabbCenter(&self) -> Point;
}


// computes the merging of a set of AABB's
// /return tuple of aabb center and extend
fn calcAabbMerge<T>(elements: &Vec<Rc<T>>) -> (Point, Vec3) where T:HasAabb {
	let mut extendMin:Vec3 = Vec3::new(std::f64::INFINITY, std::f64::INFINITY, std::f64::INFINITY);
	let mut extendMax:Vec3 = Vec3::new(-std::f64::INFINITY, -std::f64::INFINITY, -std::f64::INFINITY);
	
	for iElement in elements {
		let iMin = &iElement.retAabbCenter().p - &iElement.retAabbExtend().scale(0.5);
		let iMax = &iElement.retAabbCenter().p + &iElement.retAabbExtend().scale(0.5);

		extendMin.x = extendMin.x.min(iMin.x);
		extendMin.y = extendMin.y.min(iMin.y);
		extendMin.z = extendMin.z.min(iMin.z);

		extendMax.x = extendMax.x.max(iMax.x);
		extendMax.y = extendMax.y.max(iMax.y);
		extendMax.z = extendMax.z.max(iMax.z);
	}

	let center = (&extendMax + &extendMin).scale(0.5);
	let extend = &extendMax - &extendMin;

	(Point{p:center}, extend)
}


use std::rc::Rc;

// single polygon
// is a primitive mainly for prototyping functionality
struct PrimitivePolygon {
	vertices: [Point; 3],

	// TODO< material >
}


// generic primitive
// used for scene description where we don't know the exact type at runtime
enum Primitive {
	SPHERE(PrimitiveSphere),
	POLYGON(PrimitivePolygon)
}

impl HasAabb for Primitive {
	fn retAabbExtend(&self) -> Vec3 {
		match self {
			Primitive::SPHERE(primitiveSphere) => {primitiveSphere.retAabbExtend()}
			Primitive::POLYGON(primitivePolygon) => {
				let mut extendMin = primitivePolygon.vertices[0].p.clone();
				let mut extendMax = primitivePolygon.vertices[0].p.clone();

				extendMin.x = extendMin.x.min(primitivePolygon.vertices[1].p.x);
				extendMin.y = extendMin.y.min(primitivePolygon.vertices[1].p.y);
				extendMin.z = extendMin.z.min(primitivePolygon.vertices[1].p.z);

				extendMax.x = extendMax.x.max(primitivePolygon.vertices[1].p.x);
				extendMax.y = extendMax.y.max(primitivePolygon.vertices[1].p.y);
				extendMax.z = extendMax.z.max(primitivePolygon.vertices[1].p.z);


				extendMin.x = extendMin.x.min(primitivePolygon.vertices[2].p.x);
				extendMin.y = extendMin.y.min(primitivePolygon.vertices[2].p.y);
				extendMin.z = extendMin.z.min(primitivePolygon.vertices[2].p.z);

				extendMax.x = extendMax.x.max(primitivePolygon.vertices[2].p.x);
				extendMax.y = extendMax.y.max(primitivePolygon.vertices[2].p.y);
				extendMax.z = extendMax.z.max(primitivePolygon.vertices[2].p.z);

				&extendMax - &extendMin
			}
		}
	}

	fn retAabbCenter(&self) -> Point {
		match self {
			Primitive::SPHERE(primitiveSphere) => {primitiveSphere.retAabbCenter()}
			Primitive::POLYGON(primitivePolygon) => {
				let mut extendMin = primitivePolygon.vertices[0].p.clone();
				let mut extendMax = primitivePolygon.vertices[0].p.clone();

				extendMin.x = extendMin.x.min(primitivePolygon.vertices[1].p.x);
				extendMin.y = extendMin.y.min(primitivePolygon.vertices[1].p.y);
				extendMin.z = extendMin.z.min(primitivePolygon.vertices[1].p.z);

				extendMax.x = extendMax.x.max(primitivePolygon.vertices[1].p.x);
				extendMax.y = extendMax.y.max(primitivePolygon.vertices[1].p.y);
				extendMax.z = extendMax.z.max(primitivePolygon.vertices[1].p.z);


				extendMin.x = extendMin.x.min(primitivePolygon.vertices[2].p.x);
				extendMin.y = extendMin.y.min(primitivePolygon.vertices[2].p.y);
				extendMin.z = extendMin.z.min(primitivePolygon.vertices[2].p.z);

				extendMax.x = extendMax.x.max(primitivePolygon.vertices[2].p.x);
				extendMax.y = extendMax.y.max(primitivePolygon.vertices[2].p.y);
				extendMax.z = extendMax.z.max(primitivePolygon.vertices[2].p.z);

				Point{p:(&extendMin + &extendMax).scale(0.5)}
			}
		}
	}
}

impl HasCenter for Primitive {
	fn retCenter(&self) -> Point {
		match self {
			Primitive::SPHERE(primitiveSphere) => {primitiveSphere.retCenter()}
			Primitive::POLYGON(primitivePolygon) => {
				Point{p:(&(&primitivePolygon.vertices[0].p + &primitivePolygon.vertices[1].p) + &primitivePolygon.vertices[2].p).scale(1.0/3.0)}
			}
		}
	}
}



extern crate rand;
use rand::{thread_rng, ThreadRng, Rng};



struct SerializedBvhNode {
	extend: Vec3,
	center: Point,

	idx: i64,

	leftChildrenIdx: i64,
	rightChildrenIdx: i64,

	isLeaf: bool,

	leafElementIdx: i64, // index of BVH leaf element
}

// can be a primitive or a Element(Mesh, etc.)
struct SerializedBvhLeafElement {
	type_: i64,

	vertex0: Vector4<f64>,
	vertex1: Vector4<f64>,
	vertex2: Vector4<f64>,
}




fn serializeBvh(bvh: &Option<Rc<bvh::BvhNode<Primitive>>>) -> (Vec<SerializedBvhNode>, Vec<SerializedBvhLeafElement>) {

	fn traverseRecursivly(
		bvhNode:Rc<bvh::BvhNode<Primitive>>,
		serialized: &mut Vec<SerializedBvhNode>,
		serializedPrimitives: &mut Vec<SerializedBvhLeafElement>,
		bvhNodeidxCounter: &mut i64,
		elementIdxCounter: &mut i64
	) -> i64 {
		
	    match &(*bvhNode) {
	    	bvh::BvhNode::LEAF{bvhCenter, bvhExtend, element} => {

			    let idx = *bvhNodeidxCounter;
				*bvhNodeidxCounter+=1;


			    // fetch index and increment index for serialized primitives
			    let elementIdx = *elementIdxCounter;
			    *elementIdxCounter+=1;

			    serialized.push(SerializedBvhNode {
					extend: bvhExtend.clone(),
					center: bvhCenter.clone(),

					idx: idx,

					leftChildrenIdx: -1,
					rightChildrenIdx: -1,

					isLeaf: true,

					leafElementIdx: elementIdx,
				});

				

			    // serialize primitive/element


				match *Rc::clone(&element) {
					Primitive::SPHERE(ref primitiveSphere) => {
						serializedPrimitives.push(SerializedBvhLeafElement {
							type_: 0, // type for sphere

							vertex0: Vector4::new(primitiveSphere.pos.p.x, primitiveSphere.pos.p.y, primitiveSphere.pos.p.z, primitiveSphere.r),
							vertex1: Vector4::new(0.0, 0.0, 0.0, 0.0),
							vertex2: Vector4::new(0.0, 0.0, 0.0, 0.0),
						});
					}
					Primitive::POLYGON(ref primitivePolygon) => {
						serializedPrimitives.push(SerializedBvhLeafElement {
							type_: 1, // type for polygon

							vertex0: Vector4::new(primitivePolygon.vertices[0].p.x, primitivePolygon.vertices[0].p.y, primitivePolygon.vertices[0].p.z, 1.0),
							vertex1: Vector4::new(primitivePolygon.vertices[1].p.x, primitivePolygon.vertices[1].p.y, primitivePolygon.vertices[1].p.z, 1.0),
							vertex2: Vector4::new(primitivePolygon.vertices[2].p.x, primitivePolygon.vertices[2].p.y, primitivePolygon.vertices[2].p.z, 1.0),
						});
					}
				}

				idx
	    	}
	    	bvh::BvhNode::BRANCH {bvhCenter, bvhExtend,  left, right} => {



			    let childrenLeftIdx = traverseRecursivly(Rc::clone(&left), serialized, serializedPrimitives, bvhNodeidxCounter, elementIdxCounter);
			    let childrenRightIdx = traverseRecursivly(Rc::clone(&right), serialized, serializedPrimitives, bvhNodeidxCounter, elementIdxCounter);

			    let idx = *bvhNodeidxCounter;
				*bvhNodeidxCounter+=1;


			    serialized.push(SerializedBvhNode {
					extend: bvhExtend.clone(),
					center: bvhCenter.clone(),

					idx: idx,

					leftChildrenIdx: childrenLeftIdx,
					rightChildrenIdx: childrenRightIdx,

					isLeaf: false,

					leafElementIdx: -1,
				});

				idx
	    	}
	    }
	}

	let mut bvhNodeidxCounter = 0; // index counter used to keep track of the indices of the bvh nodes
	let mut elementIdxCounter = 0; // index counter used to keep track of indices of elements in the leaf nodes


	let mut serialized = Vec::new();
	let mut serializedPrimitives = Vec::new();

	match bvh {
		Some(bvhRootNode) => {
			// traverse recursivly to enumerate the indices
			traverseRecursivly(Rc::clone(&bvhRootNode), &mut serialized, &mut serializedPrimitives, &mut bvhNodeidxCounter, &mut elementIdxCounter);
		}
		None => {}
	}

	(serialized, serializedPrimitives)
}



/*
struct SerializedBvhNode {
	extend: Vec3,
	center: Point,
	
	idx: i64,
	
	leftChildrenIdx: i64,
	rightChildrenIdx: i64,

	isLeaf: bool,

	leafElementIdx: i64, // index of BVH leaf element
}*/


fn test_buildAndSerializeBvh() {
	fn convVec3ToGlslVec4String(v: &Vec3) -> String {
		format!("vec4({},{},{},1.0)", v.x, v.y, v.z)
	}

	fn convVec4ToGlslVec4String(v: &Vector4<f64>) -> String {
		format!("vec4({},{},{},{})", v.x, v.y, v.z, v.w)
	}

	fn convBoolToGlslInt(b: bool) -> String {
		if b {
			return "1".to_string()
		}

		"0".to_string()
	}


	// serializes the BVH nodes to GLSL sourcecode
	// used just to debug and play around with shadertoy
	fn serializeBvhNodesToGlslSource(serializableBvhNodes: &Vec<SerializedBvhNode>) {
		

		// build string for array and print
		let leftAsString = serializableBvhNodes.iter().map(|x| x.leftChildrenIdx.to_string()).collect::<Vec<_>>().join(",");
		println!("int bvhNodeChildrenLeft[] = int[{}]({});", serializableBvhNodes.len(), leftAsString);

		let rightAsString = serializableBvhNodes.iter().map(|x| x.rightChildrenIdx.to_string()).collect::<Vec<_>>().join(",");
		println!("int bvhNodeChildrenRight[] = int[{}]({});", serializableBvhNodes.len(), rightAsString);

		let isLeafAsString = serializableBvhNodes.iter().map(|x| convBoolToGlslInt(x.isLeaf)).collect::<Vec<_>>().join(",");
		println!("int bvhIsLeaf[] = int[{}]({});", serializableBvhNodes.len(), isLeafAsString);

		let bvhCentersAsString = serializableBvhNodes.iter().map(|x| convVec3ToGlslVec4String(&x.center.p)).collect::<Vec<_>>().join(",");
		println!("vec4 bvhAabbCenter[] = vec4[{}]({});", serializableBvhNodes.len(), bvhCentersAsString);

		let bvhExtendAsString = serializableBvhNodes.iter().map(|x| convVec3ToGlslVec4String(&x.extend)).collect::<Vec<_>>().join(",");
		println!("vec4 bvhAabbExtend[] = vec4[{}]({});", serializableBvhNodes.len(), bvhExtendAsString);

		let leafElementIndicesString = serializableBvhNodes.iter().map(|x| x.leafElementIdx.to_string()).collect::<Vec<_>>().join(",");
		println!("int bvhLeafNodeIndices[] = int[{}]({});", serializableBvhNodes.len(), leafElementIndicesString);

		// we need to set the root index
		println!("int bvhRootNodeIdx = {};", serializableBvhNodes.len()-1);

		println!("");
	}


	fn serializeBvhElementsToGlslSource(serializableBvhElements: &Vec<SerializedBvhLeafElement>) {
		let bvhElementTypesAsString = serializableBvhElements.iter().map(|x| x.type_.to_string()).collect::<Vec<_>>().join(",");
		println!("int bvhLeafNodeType[] = int[{}]({});", serializableBvhElements.len(), bvhElementTypesAsString);

		let bvhElementVertex0sAsString = serializableBvhElements.iter().map(|x| convVec4ToGlslVec4String(&x.vertex0)).collect::<Vec<_>>().join(",");
		println!("vec4 bvhLeafNodeVertex0[] = vec4[{}]({});", serializableBvhElements.len(), bvhElementVertex0sAsString);

		let bvhElementVertex1sAsString = serializableBvhElements.iter().map(|x| convVec4ToGlslVec4String(&x.vertex1)).collect::<Vec<_>>().join(",");
		println!("vec4 bvhLeafNodeVertex1[] = vec4[{}]({});", serializableBvhElements.len(), bvhElementVertex1sAsString);

		let bvhElementVertex2sAsString = serializableBvhElements.iter().map(|x| convVec4ToGlslVec4String(&x.vertex2)).collect::<Vec<_>>().join(",");
		println!("vec4 bvhLeafNodeVertex2[] = vec4[{}]({});", serializableBvhElements.len(), bvhElementVertex2sAsString);

		println!("");
	}



	

	let mut rng = rand::thread_rng();

	let mut scenePrimitives = Vec::new(); // all primitives of the scene which will be stored in a bvh

	// TODO< put scene into vector of primitives >
	/*
	{
		scenePrimitives.push(Rc::new(Primitive::SPHERE(PrimitiveSphere{
			id: 0,
			shading: Shading {
				colorR: 1.0,
			    colorG: 0.0,
			    colorB: 0.0,
			},

			pos: Point::new(1.0, 0.0, 5.0),
			r: 0.2
		})));
	}

	{
		scenePrimitives.push(Rc::new(Primitive::SPHERE(PrimitiveSphere{
			id: 0,
			shading: Shading {
				colorR: 1.0,
			    colorG: 0.0,
			    colorB: 0.0,
			},

			pos: Point::new(-1.0, 0.0, 5.0),
			r: 0.2
		})));
	}
	*/

	{
        // fill scene with test polygons

        for i in 0..300 {
            scenePrimitives.push(Rc::new(Primitive::POLYGON(PrimitivePolygon{
                vertices: [Point::new(1.0 - (i as f64) * 0.01, 0.0, 5.0), Point::new(-1.0 - (i as f64) * 0.01, 0.0, 5.0), Point::new(0.0 - (i as f64) * 0.01, 1.0, 5.0)],
            })));
        }



	}


	// build BVH
	let bvhRoot = bvh::splitAndCreateBvhNode(&mut scenePrimitives, &mut rng);

	// convert it to a (linearized) reoresentation which maps good to shaders as SoA's
	let (serializedBvhNodes, serializedBvhElements) = serializeBvh(&bvhRoot);

	// convert and print GLSL
	serializeBvhNodesToGlslSource(&serializedBvhNodes);
	serializeBvhElementsToGlslSource(&serializedBvhElements);
}









//////////////////////
// FPS utility

pub struct FpsMeasure {
    pub lastSystemTime: u64,
    pub lastSecondSystemTime: u64,

    pub framesInThisSecond: i64,
}

impl FpsMeasure {
    pub fn tick(&mut self) {

        let timeInNs = time::precise_time_ns();

        if self.lastSecondSystemTime + 1000000000 <= timeInNs {
            println!("fps=~{}", self.framesInThisSecond);

            self.lastSecondSystemTime = timeInNs;
            self.framesInThisSecond = 0;
        }
        else {
            self.framesInThisSecond+=1;
        }
    }
}












// TODO< remove HasCenter because it is redudant >


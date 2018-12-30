#![allow(non_snake_case)]






#[repr(C)]
struct GlslBvhNode {
    pub nodeChildrenLeft: i32,
    pub nodeChildrenRight: i32,
    pub isLeaf: i32, // bool
    pub leafNodeIdx: i32,

    pub aabbCenter: [f32; 4],
    pub aabbExtend: [f32; 4],
}

pub struct BvhNode {
	pub nodeChildrenLeft: i32,
    pub nodeChildrenRight: i32,
    pub isLeaf: bool,
    pub leafNodeIdx: i32,

    pub aabbCenter: Vector3<f64>,
    pub aabbExtend: Vector3<f64>,
}



#[repr(C)]
struct GlslBvhLeafNode {
    pub nodeType: i32,
    pub materialIdx: i32,
    pub padding0: i32,
    pub padding1: i32,

    pub vertex0: [f32; 4],
    pub vertex1: [f32; 4],
    pub vertex2: [f32; 4],
}

pub struct BvhLeafNode {
	pub nodeType: i32,
	pub materialIdx: i32,

    pub vertex0: Vector4<f64>,
    pub vertex1: Vector4<f64>,
    pub vertex2: Vector4<f64>,
}



#[repr(C)]
struct GlslMaterial {
	pub type_: i32,
	pub padding0: i32,
    pub padding1: i32,
    pub padding2: i32,

    pub baseColor: [f32; 4],
}

pub struct Material {
	pub type_: i32,
	pub baseColor: Vector3<f32>,
}


extern crate sdl2;

use nalgebra::{U4, Matrix, MatrixArray, Vector4, Vector3};

use super::FpsMeasure;


// structure to keep the whole graphics engine in one place
pub struct GraphicsEngine {
	sdl: sdl2::Sdl, // sdl handle
	window: sdl2::video::Window,
	videoSubsystem: sdl2::VideoSubsystem, // handle for sdl video subsystem

	glContext: sdl2::video::GLContext,
	pub eventPump: sdl2::EventPump,


	// used to measure the frame timing and FPS
	pub fpsMeasure: FpsMeasure,


	// pub for testing
	pub shaderProgram: Option<renderer_gl::OpenGlProgram>, // main shader program which implements the raytracer

	// pub for testing
	pub vao: Option<gl::types::GLuint>, // VAO for the full screen quad


	// pub for testing
	pub bvhNodesSsbo: Option<gl::types::GLuint>,
	pub bvhLeafNodesSsbo: Option<gl::types::GLuint>,
	pub materialsSsbo: Option<gl::types::GLuint>,
}

pub fn makeGraphicsEngine() -> Result<GraphicsEngine,String> {
	let mut eventPump;
    
    // most if not all variables have to be ept alive as long as the program is used
    let sdl;
    let window;
    let videoSubsystem;
    
    let glContext;

    { // initialize sdl and OpenGL
	    let glAttr;

	    sdl = sdl2::init().unwrap();

	    videoSubsystem = sdl.video().unwrap();

	    glAttr = videoSubsystem.gl_attr();

		glAttr.set_context_profile(sdl2::video::GLProfile::Core);
		glAttr.set_context_version(4, 3);

	    window = videoSubsystem
	        .window("Engine", 900, 700)
	        .opengl() // we need openGL support
	        //.resizable()
	        .build()
	        .unwrap();

	    glContext = window.gl_create_context().unwrap();
	    gl::load_with(|s| videoSubsystem.gl_get_proc_address(s) as *const std::os::raw::c_void);

	    eventPump = sdl.event_pump().unwrap();
    }

    Ok(GraphicsEngine {
		sdl: sdl,
		videoSubsystem: videoSubsystem,
		window: window,

		glContext: glContext,
		eventPump: eventPump,


		fpsMeasure: FpsMeasure{
			lastSystemTime: 0,
	    	lastSecondSystemTime: 0,

	    	framesInThisSecond: 0,
	    },



		shaderProgram: None,
		vao: None,
		bvhNodesSsbo: None,
		bvhLeafNodesSsbo: None,
		materialsSsbo: None,
	})
}

impl GraphicsEngine {
	// initializes everything and allocates stuff
	pub fn initAndAlloc(&mut self) {
		use std::ffi::CString;

	    unsafe {
		    gl::Viewport(0, 0, 900, 700);
		    gl::ClearColor(0.3, 0.3, 0.5, 1.0);
		}


		let vertShader = renderer_gl::OpenGlShader::fromVertSource(
		    &CString::new(include_str!("simple.vert")).unwrap()
		).unwrap();

		let fragShader = renderer_gl::OpenGlShader::fromFragSource(
		    &CString::new("#version 430 core\n".to_owned() + include_str!("entry.frag")).unwrap()
		).unwrap();

		let shaderProgram = renderer_gl::OpenGlProgram::fromShaders(
	    	&[vertShader, fragShader]
		).unwrap();

		


		shaderProgram.use_();

		self.shaderProgram = Some(shaderProgram);




		unsafe {
			// TODO< make ssbo attribute of shaderprogram and add a drop trait for it >
			let mut bvhNodesSsbo: gl::types::GLuint = 0;

			use std::mem;

			let maxNumberOfElements = 1 << 12;

			gl::GenBuffers(1, &mut bvhNodesSsbo);
			self.bvhNodesSsbo = Some(bvhNodesSsbo);
			gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, bvhNodesSsbo);
			gl::BufferData(gl::SHADER_STORAGE_BUFFER, (mem::size_of::<GlslBvhNode>() * maxNumberOfElements) as isize, 0 /* we pass NULL because we just want to declare the upload type */ as *const std::ffi::c_void, gl::DYNAMIC_COPY);
			gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 0, bvhNodesSsbo); // because it is at location 0
			gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0); // unbind
		}

		unsafe {
			// TODO< make ssbo attribute of shaderprogram and add a drop trait for it >
			let mut bvhLeafNodesSsbo: gl::types::GLuint = 0;

			use std::mem;

			let maxNumberOfElements = 1 << 12;

			gl::GenBuffers(1, &mut bvhLeafNodesSsbo);
			self.bvhLeafNodesSsbo = Some(bvhLeafNodesSsbo);
			gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, bvhLeafNodesSsbo);
			gl::BufferData(gl::SHADER_STORAGE_BUFFER, (mem::size_of::<GlslBvhLeafNode>() * maxNumberOfElements) as isize, 0 /* we pass NULL because we just want to declare the upload type */ as *const std::ffi::c_void, gl::DYNAMIC_COPY);
			gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 1, bvhLeafNodesSsbo); // because it is at location 1
			gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0); // unbind
		}


		unsafe {
			// TODO< make ssbo attribute of shaderprogram and add a drop trait for it >
			let mut materialsSsbo: gl::types::GLuint = 0;

			use std::mem;

			let maxNumberOfElements = 1 << 8;

			gl::GenBuffers(1, &mut materialsSsbo);
			self.materialsSsbo = Some(materialsSsbo);
			gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, materialsSsbo);
			gl::BufferData(gl::SHADER_STORAGE_BUFFER, (mem::size_of::<GlslMaterial>() * maxNumberOfElements) as isize, 0 /* we pass NULL because we just want to declare the upload type */ as *const std::ffi::c_void, gl::DYNAMIC_COPY);
			gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 2, materialsSsbo); // because it is at location 2
			gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0); // unbind
		}






		let graphicsApiVertices: Vec<f32> = vec![
		    // positions      // colors
		    1.0, -1.0, 0.0,   1.0, 0.0, 0.0,   // bottom right
		    -1.0, -1.0, 0.0,  0.0, 0.0, 0.0,   // bottom left
		    -1.0,  1.0, 0.0,   0.0, 1.0, 0.0,    // top

		    1.0, 1.0, 0.0,  1.0, 1.0, 0.0,   // bottom left
		    1.0, -1.0, 0.0,   1.0, 0.0, 0.0,   // bottom right
		    
		    -1.0,  1.0, 0.0,   0.0, 1.0, 0.0    // top
		];

		let mut vbo: gl::types::GLuint = 0;
		unsafe {
	    	gl::GenBuffers(1, &mut vbo);
		}
		// TODO< handling of error of buffer generation >

		unsafe {
		    gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
		    gl::BufferData(
		        gl::ARRAY_BUFFER, // target
		        (graphicsApiVertices.len() * std::mem::size_of::<f32>()) as gl::types::GLsizeiptr, // size of data in bytes
		        graphicsApiVertices.as_ptr() as *const gl::types::GLvoid, // pointer to data
		        gl::STATIC_DRAW, // usage
		    );
		    gl::BindBuffer(gl::ARRAY_BUFFER, 0); // unbind the buffer
		}

		let mut vao: gl::types::GLuint = 0;
		unsafe {
		    gl::GenVertexArrays(1, &mut vao);
		}
		self.vao = Some(vao);

		// make it current by binding it
		unsafe {
	    	gl::BindVertexArray(self.vao.unwrap());
	    
	    	gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
		}

		// specify data layout for attribute 0
		unsafe {
			gl::EnableVertexAttribArray(0); // this is "layout (location = 0)" in vertex shader
		    gl::VertexAttribPointer(
		        0, // index of the generic vertex attribute ("layout (location = 0)")
		        3, // the number of components per generic vertex attribute
		        gl::FLOAT, // data type
		        gl::FALSE, // normalized (int-to-float conversion)
		        (6 * std::mem::size_of::<f32>()) as gl::types::GLint, // stride (byte offset between consecutive attributes)
		        std::ptr::null() // offset of the first component
		    );
		}

		// specify data layout for attribute 1
		unsafe {
			gl::EnableVertexAttribArray(1);
		    gl::VertexAttribPointer(
		        1, // index of the generic vertex attribute
		        3, // the number of components per generic vertex attribute
		        gl::FLOAT,
		        gl::FALSE, // normalized (int-to-float conversion)
		        (6 * std::mem::size_of::<f32>()) as gl::types::GLint, // stride (byte offset between consecutive attributes)
		        (3 * std::mem::size_of::<f32>()) as *const gl::types::GLvoid // offset of the first component
		    );
		}

		// unbind
		unsafe {
			gl::BindBuffer(gl::ARRAY_BUFFER, 0);
		    gl::BindVertexArray(0);
		}
	}

	pub fn frame(
		&mut self,

		bvhNodes: &Vec<BvhNode>,
		bvhRootNodeIdx:i32,

		bvhLeafNodes: &Vec<BvhLeafNode>,

		materials: &Vec<Material>
	) {

        use std::ffi::CString;

        



        unsafe {
	    	gl::ClearColor(0.3, 0.3, 0.5, 1.0);
	    	gl::Clear(gl::COLOR_BUFFER_BIT);
		}



		let mut uniformLocationVertexColor = 0;
		let mut uniformLocationBvhRootNodeIdx = 0;
		let mut uniformLocationBvhLeafNodesCount = 0;


		match &self.shaderProgram {
        	Some(shaderProgram) => {
				unsafe {
					let uniformName = &CString::new("ourColor").unwrap();
					uniformLocationVertexColor = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
				}

				unsafe {
					let uniformName = &CString::new("bvhRootNodeIdx").unwrap();
					uniformLocationBvhRootNodeIdx = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
				}

				unsafe {
					let uniformName = &CString::new("bvhLeafNodesCount").unwrap();
					uniformLocationBvhLeafNodesCount = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
				}
        	}
        	None => {}
        }

		match &self.shaderProgram {
        	Some(shaderProgram) => {
        		shaderProgram.use_();
        	}
        	None => {}
        }


		unsafe {
			gl::Uniform1i(uniformLocationBvhRootNodeIdx, bvhRootNodeIdx);
		}



		// push data to SSBO
		/* commented because it is not necessary this way here
		unsafe {
			// * get block index
			let mut blockIndex: gl::types::GLuint = 0;
			{
				let ssboName = &CString::new("bvhNode").unwrap();
				blockIndex = gl::GetProgramResourceIndex(shaderProgram.retId(), gl::SHADER_STORAGE_BLOCK, ssbo.as_ptr());
			}

			// * connect the shader storage block to the SSBO: we tell the shader on which binding point it will find the SSBO
			let ssboBindingPointIndex: gl::types::GLuint = 0;
			gl::ShaderStorageBlockBinding(shaderProgram.retId(), blockIndex, ssboBindingPointIndex);

		}
		 */






		{
			let mut glslBvhLeafNodes: Vec<GlslBvhLeafNode> = Vec::new();

			// translate data to GLSL format
			for iBvhLeafNode in bvhLeafNodes {
				glslBvhLeafNodes.push(GlslBvhLeafNode{
					nodeType: iBvhLeafNode.nodeType,
					materialIdx: iBvhLeafNode.materialIdx,
	    			padding0: 0,
					padding1: 0,

					vertex0: [iBvhLeafNode.vertex0.x as f32, iBvhLeafNode.vertex0.y as f32, iBvhLeafNode.vertex0.z as f32, iBvhLeafNode.vertex0.w as f32],
					vertex1: [iBvhLeafNode.vertex1.x as f32, iBvhLeafNode.vertex1.y as f32, iBvhLeafNode.vertex1.z as f32, iBvhLeafNode.vertex1.w as f32],
					vertex2: [iBvhLeafNode.vertex2.x as f32, iBvhLeafNode.vertex2.y as f32, iBvhLeafNode.vertex2.z as f32, iBvhLeafNode.vertex2.w as f32],
				});

				println!("<{},{},{},{}>", iBvhLeafNode.vertex0.x as f32, iBvhLeafNode.vertex0.y as f32, iBvhLeafNode.vertex0.z as f32, iBvhLeafNode.vertex0.w as f32);

			}


			unsafe {
				gl::Uniform1i(uniformLocationBvhLeafNodesCount, glslBvhLeafNodes.len() as i32);
			}

			// update SSBO
			unsafe {
				// copy the data to the SSBO
				gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, self.bvhLeafNodesSsbo.unwrap());
				gl::BufferSubData(
					gl::SHADER_STORAGE_BUFFER,
					0,
					(std::mem::size_of::<GlslBvhLeafNode>() * glslBvhLeafNodes.len()) as isize,
					glslBvhLeafNodes.as_mut_ptr() as *const std::ffi::c_void
				);
				gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0); // unbind
			}
		}


		// update SSBO
		unsafe {
			let mut glslMaterials: Vec<GlslMaterial> = Vec::new();

			// translate data to GLSL format
			for iMaterial in materials {
				glslMaterials.push(GlslMaterial{
					type_: iMaterial.type_,
					padding0: 0,
					padding1: 0,
					padding2: 0,

					baseColor: [iMaterial.baseColor.x as f32, iMaterial.baseColor.y as f32, iMaterial.baseColor.z as f32, 1.0],
				});
			}

			// copy the data to the SSBO
			gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, self.materialsSsbo.unwrap());
			gl::BufferSubData(
				gl::SHADER_STORAGE_BUFFER,
				0,
				(std::mem::size_of::<GlslMaterial>() * glslMaterials.len()) as isize,
				glslMaterials.as_mut_ptr() as *const std::ffi::c_void
			);
			gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0); // unbind
		}


		/* commented because this is wrong/not necessary
		unsafe {
			// * get block index
			let mut blockIndex: gl::types::GLuint = 0;
			{
				let ssboName = &CString::new("bvhLeafNode").unwrap();
				blockIndex = gl::GetProgramResourceIndex(shaderProgram.retId(), gl::SHADER_STORAGE_BLOCK, ssboName.as_ptr());
			}

			// * connect the shader storage block to the SSBO: we tell the shader on which binding point it will find the SSBO
			let ssboBindingPointIndex: gl::types::GLuint = 1;
			gl::ShaderStorageBlockBinding(shaderProgram.retId(), blockIndex, ssboBindingPointIndex);

		}
		*/



		// bind SSBO's
		unsafe {
			gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 0, self.bvhNodesSsbo.unwrap());
			gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 1, self.bvhLeafNodesSsbo.unwrap());
			gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 2, self.materialsSsbo.unwrap());
		}







		unsafe {
			// commented because it is the not vectorized version
			// we keep it here as code reference!
			//gl::Uniform4f(uniformLocationVertexColor, 0.0f32, 1.0f32, 0.0f32, 1.0f32);

			let uniformVector:Vec<f32> = vec![
				0.0f32, 1.0f32, 0.0f32, 0.0f32
			];
			let ptr = uniformVector.as_ptr() as *const gl::types::GLfloat;
			gl::Uniform4fv(uniformLocationVertexColor, (uniformVector.len() as i32)/4, ptr);
		}



		unsafe {
		    gl::BindVertexArray(self.vao.unwrap());
		    gl::DrawArrays(
		        gl::TRIANGLES, // mode
		        0, // starting index in the enabled arrays
		        6 // number of indices to be rendered
		    );
		}

		self.window.gl_swap_window();

		self.fpsMeasure.tick();
	}
}




extern crate gl;

pub mod renderer_gl {
    use gl;
    use std;
    use std::ffi::{CString, CStr};

	fn shaderFromSource(
	    source: &CStr, // modified
	    kind: gl::types::GLenum
	) -> Result<gl::types::GLuint, String> {
	    let id = unsafe { gl::CreateShader(kind) };

	    unsafe {
	    	gl::ShaderSource(id, 1, &source.as_ptr(), std::ptr::null());
	    	gl::CompileShader(id);
		}

		let mut success: gl::types::GLint = 1;
		unsafe {
	    	gl::GetShaderiv(id, gl::COMPILE_STATUS, &mut success);
		}

		if success == 0 {
	    	let mut len: gl::types::GLint = 0;
			unsafe {
	    		gl::GetShaderiv(id, gl::INFO_LOG_LENGTH, &mut len);
			}

			let error: CString = createWhitespaceCstringWithLen(len as usize);

			unsafe {
			    gl::GetShaderInfoLog(
			        id,
			        len,
			        std::ptr::null_mut(),
			        error.as_ptr() as *mut gl::types::GLchar
			    );
			}

			return Err(error.to_string_lossy().into_owned());
		}

		Ok(id)
	}

	fn createWhitespaceCstringWithLen(len: usize) -> CString {
	    // allocate buffer of correct size
	    let mut buffer: Vec<u8> = Vec::with_capacity(len + 1);
	    // fill it with len spaces
	    buffer.extend([b' '].iter().cycle().take(len));
	    // convert buffer to CString
	    unsafe { CString::from_vec_unchecked(buffer) }
	}

	pub struct OpenGlShader {
	    id: gl::types::GLuint,
	}
	impl OpenGlShader {
	    fn fromSource(
	        source: &CStr,
	        kind: gl::types::GLenum
	    ) -> Result<OpenGlShader, String> {
	        let id = shaderFromSource(source, kind)?;
	        Ok(OpenGlShader { id })
	    }
	    
	    pub fn fromVertSource(source: &CStr) -> Result<OpenGlShader, String> {
	        OpenGlShader::fromSource(source, gl::VERTEX_SHADER)
	    }

	    pub fn fromFragSource(source: &CStr) -> Result<OpenGlShader, String> {
	        OpenGlShader::fromSource(source, gl::FRAGMENT_SHADER)
	    }

	    pub fn retId(&self) -> gl::types::GLuint {
	        self.id
	    }
	}
	impl Drop for OpenGlShader {
	    fn drop(&mut self) {
	        unsafe {
	            gl::DeleteShader(self.id);
	        }
	    }
	}




	pub struct OpenGlProgram {
	    id: gl::types::GLuint,
	}

	impl OpenGlProgram {
	    pub fn fromShaders(shaders: &[OpenGlShader]) -> Result<OpenGlProgram, String> {
	        let program_id = unsafe { gl::CreateProgram() };

	        for shader in shaders {
	            unsafe { gl::AttachShader(program_id, shader.retId()); }
	        }

	        unsafe { gl::LinkProgram(program_id); }

	        { // error handling
				let mut success: gl::types::GLint = 1;
				unsafe {
				    gl::GetProgramiv(program_id, gl::LINK_STATUS, &mut success);
				}

				if success == 0 {
				    let mut len: gl::types::GLint = 0;
				    unsafe {
				        gl::GetProgramiv(program_id, gl::INFO_LOG_LENGTH, &mut len);
				    }

				    let error = createWhitespaceCstringWithLen(len as usize);

				    unsafe {
				        gl::GetProgramInfoLog(
				            program_id,
				            len,
				            std::ptr::null_mut(),
				            error.as_ptr() as *mut gl::types::GLchar
				        );
				    }

				    return Err(error.to_string_lossy().into_owned());
				}
	        }

	        // we need to detach because drop doesn't detach!
	        for shader in shaders {
	            unsafe { gl::DetachShader(program_id, shader.retId()); }
	        }

	        Ok(OpenGlProgram { id: program_id })
	    }

	    pub fn use_(&self) {
    		unsafe {
        		gl::UseProgram(self.id);
    		}
		}

	    pub fn retId(&self) -> gl::types::GLuint {
	        self.id
	    }
	}

	impl Drop for OpenGlProgram {
	    fn drop(&mut self) {
	        unsafe {
	            gl::DeleteProgram(self.id);
	        }
	    }
	}










	pub struct OpenGlTexture {
	    id: gl::types::GLuint,

	    type_: gl::types::GLuint,
	}

	
	
	impl OpenGlTexture {
		// /param type_ OpenGL type of the texture, for example gl::TEXTURE_RECTANGLE or gl::TEXTURE_2D
	    pub fn make(type_: gl::types::GLuint, width: i32, height: i32) -> Result<OpenGlTexture, String> {
	    	let mut id_: gl::types::GLuint = 0;
	    	//let type_: gl::types::GLuint = gl::TEXTURE_RECTANGLE;//gl::TEXTURE_2D;
	    	
	    	unsafe {
	    		gl::GenTextures(1, &mut id_);
	    	}

	    	unsafe {
				// "Bind" the newly created texture : all future texture functions will modify this texture
				gl::BindTexture(type_, id_);

				// Give the image format to OpenGL
				gl::TexImage2D(
					type_, // target
					0, // type

					// or R32F
					gl::RGBA32F as i32, // internalformat
					width as gl::types::GLint,
					height as gl::types::GLint,
					0,
					gl::RGBA,
					gl::UNSIGNED_BYTE,
					0 as *const std::ffi::c_void
				);

				gl::TexParameteri(type_, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
				gl::TexParameteri(type_, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
			}

	        Ok(OpenGlTexture{id: id_, type_: type_})
	    }

	    pub fn bind(&self) {
    		unsafe {
        		gl::BindTexture(self.type_, self.id);
    		}
		}

	    pub fn retId(&self) -> gl::types::GLuint {
	        self.id
	    }
	}

	impl Drop for OpenGlTexture {
	    fn drop(&mut self) {
	        unsafe {
	            gl::DeleteTextures(1, &self.id);
	        }
	    }
	}
	

}
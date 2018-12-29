#![allow(non_snake_case)]







#[repr(C)]
pub struct GlslBvhNode {
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


extern crate sdl2;

use nalgebra::{U4, Matrix, MatrixArray, Vector4, Vector3};

pub fn openglMain(
	bvhNodes: &Vec<BvhNode>,
	bvhRootNodeIdx:i32,


	bvhLeafNodeType:&Vec<i32>,
	bvhLeafNodeVertex0:&Vec<Vector4<f64>>,
	bvhLeafNodeVertex1:&Vec<Vector4<f64>>,
	bvhLeafNodeVertex2:&Vec<Vector4<f64>>
) {
    
    let mut eventPump;
    
    // most if not all variables have to be ept alive as long as the program is used
    let sdl;
    let window;
    let videoSubsystem;
    let glAttr;
    let glContext;
    let gl;

    { // initialize sdl and OpenGL
	    sdl = sdl2::init().unwrap();
	    videoSubsystem = sdl.video().unwrap();

	    glAttr = videoSubsystem.gl_attr();

		glAttr.set_context_profile(sdl2::video::GLProfile::Core);
		glAttr.set_context_version(4, 3);

	    window = videoSubsystem
	        .window("Game", 900, 700)
	        .opengl() // we need openGL support
	        .resizable()
	        .build()
	        .unwrap();

	    glContext = window.gl_create_context().unwrap();
	    gl = gl::load_with(|s| videoSubsystem.gl_get_proc_address(s) as *const std::os::raw::c_void);

	    eventPump = sdl.event_pump().unwrap();
    }



    unsafe {
	    gl::Viewport(0, 0, 900, 700);
	    gl::ClearColor(0.3, 0.3, 0.5, 1.0);
	}


	
	use std::ffi::CString;

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

	unsafe {
		let mut glslBvhNodes: Vec<GlslBvhNode> = Vec::new();

		let mut ssbo: gl::types::GLuint = 0;
		gl::GenBuffers(1, &mut ssbo);
		gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, ssbo);
		gl::BufferData(gl::SHADER_STORAGE_BUFFER, ((/* sizeof(shader_data) */ 4*4 + 4*4 + 4*4) * glslBvhNodes.len()) as isize, glslBvhNodes.as_ptr() as *const std::ffi::c_void, gl::DYNAMIC_COPY);
		gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0); // unbind
	}






	let graphicsApiVertices: Vec<f32> = vec![
	    // positions      // colors
	    0.5, -0.5, 0.0,   1.0, 0.0, 0.0,   // bottom right
	    -0.5, -0.5, 0.0,  0.0, 0.0, 0.0,   // bottom left
	    -0.5,  0.5, 0.0,   0.0, 1.0, 0.0,    // top

	    0.5, 0.5, 0.0,  1.0, 1.0, 0.0,   // bottom left
	    0.5, -0.5, 0.0,   1.0, 0.0, 0.0,   // bottom right
	    
	    -0.5,  0.5, 0.0,   0.0, 1.0, 0.0    // top
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

	// make it current by binding it
	unsafe {
    	gl::BindVertexArray(vao);
    
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






    'main: loop {
        for event in eventPump.poll_iter() {
            match event {
                sdl2::event::Event::Quit {..} => break 'main,
                _ => {},
            }
        }

        unsafe {
	    	gl::ClearColor(0.3, 0.3, 0.5, 1.0);
	    	gl::Clear(gl::COLOR_BUFFER_BIT);
		}


		let uniformLocationVertexColor;
		unsafe {
			let uniformName = &CString::new("ourColor").unwrap();
			uniformLocationVertexColor = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
		}

		//let uniformLocationBvhNodeChildrenLeft;
		//let uniformLocationBvhNodeChildrenRight;
		//let uniformLocationBvhIsLeaf;
		//let uniformLocationBvhAabbCenter;
		//let uniformLocationBvhAabbExtend;
		//let uniformLocationBvhLeafNodeIndices;
		let uniformLocationBvhRootNodeIdx;

		let uniformLocationBvhLeafNodeType;
		let uniformLocationBvhLeafNodeVertex0;
		let uniformLocationBvhLeafNodeVertex1;
		let uniformLocationBvhLeafNodeVertex2;

		/*
		unsafe {
			let uniformName = &CString::new("bvhNodeChildrenLeft").unwrap();
			uniformLocationBvhNodeChildrenLeft = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
		}

		unsafe {
			let uniformName = &CString::new("bvhNodeChildrenRight").unwrap();
			uniformLocationBvhNodeChildrenRight = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
		}

		unsafe {
			let uniformName = &CString::new("bvhIsLeaf").unwrap();
			uniformLocationBvhIsLeaf = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
		}

		unsafe {
			let uniformName = &CString::new("bvhAabbCenter").unwrap();
			uniformLocationBvhAabbCenter = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
		}

		unsafe {
			let uniformName = &CString::new("bvhAabbExtend").unwrap();
			uniformLocationBvhAabbExtend = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
		}

		unsafe {
			let uniformName = &CString::new("bvhLeafNodeIndices").unwrap();
			uniformLocationBvhLeafNodeIndices = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
		}
		*/

		unsafe {
			let uniformName = &CString::new("bvhRootNodeIdx").unwrap();
			uniformLocationBvhRootNodeIdx = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
		}






		unsafe {
			let uniformName = &CString::new("bvhLeafNodeType").unwrap();
			uniformLocationBvhLeafNodeType = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
		}

		unsafe {
			let uniformName = &CString::new("bvhLeafNodeVertex0").unwrap();
			uniformLocationBvhLeafNodeVertex0 = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
		}

		unsafe {
			let uniformName = &CString::new("bvhLeafNodeVertex1").unwrap();
			uniformLocationBvhLeafNodeVertex1 = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
		}

		unsafe {
			let uniformName = &CString::new("bvhLeafNodeVertex2").unwrap();
			uniformLocationBvhLeafNodeVertex2 = gl::GetUniformLocation(shaderProgram.retId(), uniformName.as_ptr());
		}













		shaderProgram.use_();

		/*
		unsafe {
			let uniformVector:&Vec<i32> = &bvhNodeChildrenLeft;
			let ptr = uniformVector.as_ptr() as *const gl::types::GLint;
			gl::Uniform1iv(uniformLocationBvhNodeChildrenLeft, uniformVector.len() as i32, ptr);
		}

		unsafe {
			let uniformVector:&Vec<i32> = &bvhNodeChildrenRight;
			let ptr = uniformVector.as_ptr() as *const gl::types::GLint;
			gl::Uniform1iv(uniformLocationBvhNodeChildrenRight, uniformVector.len() as i32, ptr);
		}

		unsafe {
			let uniformVector:&Vec<i32> = &bvhIsLeaf;
			let ptr = uniformVector.as_ptr() as *const gl::types::GLint;
			gl::Uniform1iv(uniformLocationBvhIsLeaf, uniformVector.len() as i32, ptr);
		}

		unsafe {
			let mut uniformVector:Vec<f32> = Vec::new();
			for i in bvhAabbCenter {
				uniformVector.push(i.x as f32);
				uniformVector.push(i.y as f32);
				uniformVector.push(i.z as f32);
				uniformVector.push(1.0f32);
			}


			let ptr = uniformVector.as_ptr() as *const gl::types::GLfloat;
			gl::Uniform4fv(uniformLocationBvhAabbCenter, (uniformVector.len() as i32)/4, ptr);
		}

		unsafe {
			let mut uniformVector:Vec<f32> = Vec::new();
			for i in bvhAabbExtend {
				uniformVector.push(i.x as f32);
				uniformVector.push(i.y as f32);
				uniformVector.push(i.z as f32);
				uniformVector.push(1.0f32);
			}


			let ptr = uniformVector.as_ptr() as *const gl::types::GLfloat;
			gl::Uniform4fv(uniformLocationBvhAabbExtend, (uniformVector.len() as i32)/4, ptr);
		}

		unsafe {
			let uniformVector:&Vec<i32> = &bvhLeafNodeIndices;
			let ptr = uniformVector.as_ptr() as *const gl::types::GLint;
			gl::Uniform1iv(uniformLocationBvhLeafNodeIndices, uniformVector.len() as i32, ptr);
		}
		*/

		unsafe {
			gl::Uniform1i(uniformLocationBvhRootNodeIdx, bvhRootNodeIdx);
		}



		unsafe {
			let uniformVector:&Vec<i32> = &bvhLeafNodeType;
			let ptr = uniformVector.as_ptr() as *const gl::types::GLint;
			gl::Uniform1iv(uniformLocationBvhLeafNodeType, uniformVector.len() as i32, ptr);
		}

		unsafe {
			let mut uniformVector:Vec<f32> = Vec::new();
			for i in bvhLeafNodeVertex0 {
				uniformVector.push(i.x as f32);
				uniformVector.push(i.y as f32);
				uniformVector.push(i.z as f32);
				uniformVector.push(i.w as f32);
			}

			let ptr = uniformVector.as_ptr() as *const gl::types::GLfloat;
			gl::Uniform4fv(uniformLocationBvhLeafNodeVertex0, (uniformVector.len() as i32)/4, ptr);
		}

		unsafe {
			let mut uniformVector:Vec<f32> = Vec::new();
			for i in bvhLeafNodeVertex1 {
				uniformVector.push(i.x as f32);
				uniformVector.push(i.y as f32);
				uniformVector.push(i.z as f32);
				uniformVector.push(i.w as f32);
			}

			let ptr = uniformVector.as_ptr() as *const gl::types::GLfloat;
			gl::Uniform4fv(uniformLocationBvhLeafNodeVertex1, (uniformVector.len() as i32)/4, ptr);
		}

		unsafe {
			let mut uniformVector:Vec<f32> = Vec::new();
			for i in bvhLeafNodeVertex2 {
				uniformVector.push(i.x as f32);
				uniformVector.push(i.y as f32);
				uniformVector.push(i.z as f32);
				uniformVector.push(i.w as f32);
			}

			let ptr = uniformVector.as_ptr() as *const gl::types::GLfloat;
			gl::Uniform4fv(uniformLocationBvhLeafNodeVertex2, (uniformVector.len() as i32)/4, ptr);
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
		    gl::BindVertexArray(vao);
		    gl::DrawArrays(
		        gl::TRIANGLES, // mode
		        0, // starting index in the enabled arrays
		        6 // number of indices to be rendered
		    );
		}

		window.gl_swap_window();
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
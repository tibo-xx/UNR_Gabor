/*
* File name: Shader.h
* Created by Roger Hoang.
*/

#ifndef __SHADER_H__
#define __SHADER_H__

#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <map>
#include <list>
#include <cstring>

class Shader
{
public:
	inline void addShader(const char* path, GLenum shaderType)
	{
		GLchar* shader = _readShader(path);
		if (!shader)
			return;
		_objects.push_back(std::pair<char*,GLenum>(shader, shaderType));
	}

	inline void setParameter(GLenum param, GLenum v)
	{
		_parameters.push_back(std::pair<GLenum, GLenum>(param, v));
	}

	inline void compile()
	{
		GLuint program = glCreateProgram();
		for (std::list<std::pair<char*,GLenum> >::iterator it = _objects.begin();
		     it != _objects.end(); it++)
		{
			GLuint shader = glCreateShader(it->second);
			const GLchar* source = it->first;
			glShaderSource(shader, 1, &source, NULL);
			glCompileShader(shader);
			int status = 0;
			glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
			int logLength = 0;
			glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
			if (logLength > 1)
			{
				char* buf = new char[logLength];
				glGetShaderInfoLog(shader, logLength, NULL, buf);
				printf("%s\n", buf);
				delete [] buf;
			}
			if (status == GL_FALSE)
			{
				exit(-1);
			}
			glAttachObjectARB(program, shader);
		}
#ifdef G80
		for (std::list<std::pair<GLenum, GLenum> >::iterator it = _parameters.begin();
		     it != _parameters.end(); it++)
		{
			glProgramParameteriEXT(program, it->first, it->second);
		}
#endif
		glLinkProgram(program);

		int status = 0;
		glGetProgramiv(program, GL_LINK_STATUS, &status);

		int logLength = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

		if (logLength != 0)
		{
			char* buf = new char[logLength];
			glGetProgramInfoLog(program, logLength, NULL, buf);
			printf("%s\n", buf);
			delete [] buf;
		}

		if (status == GL_FALSE)
		{
			exit(-1);
		}

		_id = program;

		for (std::list<std::pair<char*,GLenum> >::iterator it = _objects.begin();
			 it != _objects.end(); it++)
		{
			delete [] it->first;
		}
		_objects.clear();
	}

	inline void begin() const
	{
		glUseProgram(_id);
	}

	inline void end() const
	{
		glUseProgram(0);
	}

	inline void set(const char* param, int v1) const
	{
		if (-1 == glGetUniformLocation(_id, param))
			printf("shit %s\n", param);
		glUniform1i(glGetUniformLocation(_id, param), v1);
	}
	inline void set(const char* param, int v1, int v2) const
	{
		glUniform2i(glGetUniformLocation(_id, param), v1, v2);
	}
	inline void set(const char* param, int v1, int v2, int v3) const
	{
		glUniform3i(glGetUniformLocation(_id, param), v1, v2, v3);
	}
	inline void set(const char* param, int v1, int v2, int v3, int v4) const
	{
		glUniform4i(glGetUniformLocation(_id, param), v1, v2, v3, v4);
	}
	inline void set(const char* param, float v1) const
	{
		glUniform1f(glGetUniformLocation(_id, param), v1);
	}
	inline void set(const char* param, float v1, float v2) const
	{
		glUniform2f(glGetUniformLocation(_id, param), v1, v2);
	}
	inline void set(const char* param, float v1, float v2, float v3) const
	{
		glUniform3f(glGetUniformLocation(_id, param), v1, v2, v3);
	}
	inline void set(const char* param, float v1, float v2, float v3, float v4) const
	{
		glUniform4f(glGetUniformLocation(_id, param), v1, v2, v3, v4);
	}

	GLuint id()
	{
		return _id;
	}
	
protected:
	std::list<std::pair<GLchar*,GLenum> > _objects;
	std::list<std::pair<GLenum, GLenum> > _parameters;
	GLuint _id;

	inline GLchar* _readShader(const char* path)
	{  
		long size;
		GLchar* buf;
		FILE* fp = fopen(path,"rb");
		if (!fp)
		{
			printf("Error: Could not open shader %s\n", path);
			return NULL;
		}
		fseek(fp,0,SEEK_END);
		size = ftell(fp);
		buf = new GLchar[size + 1];
		memset(buf, 0, size + 1);
		fseek(fp,0,SEEK_SET);
		fread(buf, 1, size, fp);
		fclose(fp);
		return buf;
	}
};

#endif 


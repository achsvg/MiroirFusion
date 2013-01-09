varying vec3 normal;
varying vec3 lightDir;
varying vec3 eyeVec;
varying vec2 textCoord;
varying float att;

void main()
{
	normal = gl_NormalMatrix * gl_Normal;
	vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);
	lightDir = vec3(gl_LightSource[0].position - vVertex);
	eyeVec = -vVertex;
	
	float d = length(lightDir);
	
	att = 1.0 / ( gl_LightSource[0].constantAttenuation + 
		(gl_LightSource[0].linearAttenuation*d) + 
		(gl_LightSource[0].quadraticAttenuation*d*d) );
	
	textCoord = vec2( gl_MultiTexCoord0 );
	
	gl_Position = ftransform();
}
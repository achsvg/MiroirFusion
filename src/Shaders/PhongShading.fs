varying vec3 normal;
varying vec3 lightDir;
varying vec2 textCoord; 
varying vec3 eyeVec;
varying float att;
uniform sampler2D texture;	
		
void main()
{
	vec4 textColor = texture2D( texture,textCoord );
	vec4 finalColor = ( gl_FrontLightModelProduct.sceneColor * gl_FrontMaterial.ambient ) + 
					  ( gl_LightSource[0].ambient * gl_FrontMaterial.ambient * att );
	float alpha = gl_FrontMaterial.diffuse.a * textColor.a;
	
	vec3 N = normalize(normal);
	vec3 L = normalize(lightDir);
	
	float NdotL = max( dot( N, L ), 0.0 );
	
	if(NdotL > 0.0)
	{
		finalColor += gl_LightSource[0].diffuse * 
		               gl_FrontMaterial.diffuse * 
					   NdotL * att;	
		
		vec3 E = normalize(eyeVec);
		vec3 R = normalize( -reflect(L, N) );
		float specular = pow( max(dot(R, E), 0.0), 
		                 gl_FrontMaterial.shininess );
		finalColor += gl_LightSource[0].specular * 
		              gl_FrontMaterial.specular * 
					  specular * att;
	}
	finalColor *= textColor;
	gl_FragColor = vec4( finalColor.xyz,alpha );	
}
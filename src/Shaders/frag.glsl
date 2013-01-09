varying vec2 texCoord;
uniform sampler2D texture;

void main()
{
	//gl_FragColor = vec4(1.0,0.0,0.5,0.5);
	gl_FragColor = texture2D(texture, texCoord);
}


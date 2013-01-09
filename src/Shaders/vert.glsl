varying vec2 texCoord;

void main()                    
{
	gl_Position = ftransform();
	
	vec2 inPos = sign(gl_Vertex.xy);
	
	texCoord = (vec2(inPos.x, -inPos.y) + 1.0)/2.0;
}
uniform sampler2D filteredTexture;
uniform float scale;
uniform float opacity;

void main()
{
	gl_FragColor = texture2D(filteredTexture, gl_TexCoord[0].xy).xxxx * scale;
	gl_FragColor.w = opacity;
	//gl_FragColor.xw = vec2(0.0,1.0);
}

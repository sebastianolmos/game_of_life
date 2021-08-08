#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

// texture sampler
uniform sampler2D texture1;

void main()
{
	vec4 tColor = texture(texture1, TexCoord);
	FragColor = vec4(tColor.g, tColor.b, tColor.g, 1.0f);
}
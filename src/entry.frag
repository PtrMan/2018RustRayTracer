

in VS_OUTPUT {
    vec3 Color;
} IN;

out vec4 Color;

uniform vec4 ourColor;

void main() {
    Color = vec4(IN.Color, 1.0f);
    //Color = ourColor;
}
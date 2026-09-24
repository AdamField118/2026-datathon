attribute vec3 vert_pos;

void main(void) {
		gl_Position = vec4(vert_pos, 1);
}

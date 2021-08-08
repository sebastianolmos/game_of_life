#pragma once
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Controller {
public:
    Controller(float scaleSpeed, float trSpeed, float iterSpeed);
    void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    void updateParams(float deltaTime);
    void setTranslate(glm::vec3* translate) {
        m_translate = translate;
    }
    void setScale(float* scale) {
        m_scale = scale;
    }
    void setIterPerSecond(float* ips) {
        m_iterPerSecond = ips;
    }

private:
    glm::vec3* m_translate;
    float* m_scale;
    float* m_iterPerSecond;
    bool m_is_w_pressed;
    bool m_is_s_pressed;
    bool m_is_a_pressed;
    bool m_is_d_pressed;
    bool m_is_z_pressed;
    bool m_is_x_pressed;
    bool m_is_left_pressed;
    bool m_is_right_pressed;
    float m_translateSpeed;
    float m_zoomSpeed;
    float m_iterSpeed;

};

Controller::Controller(float scaleSpeed, float trSpeed, float iterSpeed) {
    m_zoomSpeed = scaleSpeed;
    m_translateSpeed = trSpeed;
    m_iterSpeed = iterSpeed;
    m_is_w_pressed = false;
    m_is_s_pressed = false;
    m_is_a_pressed = false;
    m_is_d_pressed = false;
    m_is_z_pressed = false;
    m_is_x_pressed = false;
    m_is_left_pressed = false;
    m_is_right_pressed = false;
}

void Controller::updateParams(float deltaTime) {
    if (m_is_w_pressed) {
        m_translate->y -= deltaTime * m_translateSpeed;
    }
    else if (m_is_s_pressed)
    {
        m_translate->y += deltaTime * m_translateSpeed;
    }

    if (m_is_a_pressed) {
        m_translate->x += deltaTime * m_translateSpeed;
    }
    else if (m_is_d_pressed)
    {
        m_translate->x -= deltaTime * m_translateSpeed;
    }

    if (m_is_z_pressed) {
        *m_scale -= deltaTime * m_zoomSpeed;
    }
    else if (m_is_x_pressed)
    {
        *m_scale += deltaTime * m_zoomSpeed;;
    }

    if (m_is_left_pressed) {
        if (*m_iterPerSecond > 1.0f) {
            *m_iterPerSecond -= deltaTime * m_iterSpeed;
        }
        else {
            *m_iterPerSecond = 1.0f;
        }
    }
    else if (m_is_right_pressed)
    {
        if (*m_iterPerSecond < 144.0f) {
            *m_iterPerSecond += deltaTime * m_iterSpeed;
        }
        else {
            *m_iterPerSecond = 144.0f;
        }
    }
}

void Controller::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            break;
        case GLFW_KEY_SPACE:
            *m_translate = glm::vec3(0.0f);
            *m_scale = 1.0f;
            break;
        case GLFW_KEY_W:
            m_is_w_pressed = true;
            break;
        case GLFW_KEY_S:
            m_is_s_pressed = true;
            break;
        case GLFW_KEY_A:
            m_is_a_pressed = true;
            break;
        case GLFW_KEY_D:
            m_is_d_pressed = true;
            break;
        case GLFW_KEY_Z:
            m_is_z_pressed = true;
            break;
        case GLFW_KEY_X:
            m_is_x_pressed = true;
            break;
        case GLFW_KEY_LEFT:
            m_is_left_pressed = true;
            break;
        case GLFW_KEY_RIGHT:
            m_is_right_pressed = true;
            break;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        switch (key) {
        case GLFW_KEY_W:
            m_is_w_pressed = false;
            break;
        case GLFW_KEY_S:
            m_is_s_pressed = false;
            break;
        case GLFW_KEY_A:
            m_is_a_pressed = false;
            break;
        case GLFW_KEY_D:
            m_is_d_pressed = false;
            break;
        case GLFW_KEY_Z:
            m_is_z_pressed = false;
            break;
        case GLFW_KEY_X:
            m_is_x_pressed = false;
            break;
        case GLFW_KEY_LEFT:
            m_is_left_pressed = false;
            break;
        case GLFW_KEY_RIGHT:
            m_is_right_pressed = false;
            break;
        }
    }
}
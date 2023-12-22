
#include "kernel.hpp"
#include "raylib.h"
#include <iostream>

//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(void)
{
    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 1920;
    const int screenHeight = 1080;

    InitWindow(screenWidth, screenHeight, "raylib [core] example - 3d camera free");

    // Define the camera to look into our 3d world
    Camera3D camera = {0};
    camera.position = (Vector3){10.0f, 10.0f, 10.0f}; // Camera position
    camera.target = (Vector3){0.0f, 1.0f, 0.0f};      // Camera looking at point
    camera.up = (Vector3){0.0f, 1.0f, 0.0f};          // Camera up vector (rotation towards target)
    camera.fovy = 45.0f;                              // Camera field-of-view Y
    camera.projection = CAMERA_PERSPECTIVE;           // Camera projection type

    Vector3 cubePosition = {0.0f, 0.0f, 0.0f};

    Model m = LoadModel("bunnysimple.obj");
    Mesh bunny = m.meshes[0];
    Vector3 bpos = {0.0f, 0.0f, 0.0f};
    Color bcol = WHITE;
    float bscale = 1.0f;
    GPUMesh *gpubunny = to_gpu(&bunny);

    SetTargetFPS(60); // Set our game to run at 60 frames-per-second
    //--------------------------------------------------------------------------------------
    // Main game loop
    while (!WindowShouldClose()) // Detect window close button or ESC key
    {
        // Update
        //----------------------------------------------------------------------------------
        UpdateCamera(&camera, CAMERA_ORBITAL);

        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

        bunny_test_wobble(gpubunny, GetTime() * 5, 1.0f);
        from_gpu(gpubunny, &bunny);
        UpdateMeshBuffer(m.meshes[0], 0, bunny.vertices, bunny.vertexCount * sizeof(float) * 3, 0);
        // printf("%f %f %f\n", m.meshes[0].vertices[0], m.meshes[0].vertices[1], m.meshes[0].vertices[2]);

        DrawModel(m, bpos, bscale, bcol);
        DrawModelWires(m, bpos, bscale, BLACK);
        DrawGrid(10, 1.0f);

        // for (int i = 0; i < bunny.vertexCount; i++)
        // {
        //     Vector3 p;
        //     p.x = bunny.vertices[i * 3];
        //     p.y = bunny.vertices[i * 3 + 1];
        //     p.z = bunny.vertices[i * 3 + 2];
        //     DrawSphereEx(p, 0.05f, 3, 3, RED);
        // }

        EndMode3D();

        DrawRectangle(10, 10, 320, 93, Fade(SKYBLUE, 0.5f));
        DrawRectangleLines(10, 10, 320, 93, BLUE);

        DrawText("Free camera default controls:", 20, 20, 10, BLACK);
        DrawText(TextFormat("%f", GetTime()), 10, 100, 20, BLACK);

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    CloseWindow(); // Close window and OpenGL context
    UnloadModel(m);
    //--------------------------------------------------------------------------------------
    return 0;
}

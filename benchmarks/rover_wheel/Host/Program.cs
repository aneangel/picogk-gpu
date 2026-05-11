// SPDX-FileCopyrightText: Copyright (c) 2026 The picogk-gpu Authors
// SPDX-License-Identifier: Apache-2.0

using Leap71.RoverExamples;
using PicoGK;

float voxelSize = 0.5f;
string mode = "preset";

string? envVox = Environment.GetEnvironmentVariable("PICOGK_VOXEL_SIZE");
if (!string.IsNullOrEmpty(envVox) && float.TryParse(envVox, out float vEnv))
{
    voxelSize = vEnv;
}
else if (args.Length >= 1 && float.TryParse(args[0], out float vArg))
{
    voxelSize = vArg;
}

string? envMode = Environment.GetEnvironmentVariable("PICOGK_ROVERWHEEL_MODE");
if (!string.IsNullOrEmpty(envMode))
{
    mode = envMode;
}
else if (args.Length >= 2)
{
    mode = args[1];
}

Console.WriteLine($"[picogk-gpu/bench/RoverWheel] voxel_size={voxelSize} mode={mode}");

System.Threading.ThreadStart task = mode == "random"
    ? new System.Threading.ThreadStart(WheelShowCase.RandomWheelTask)
    : new System.Threading.ThreadStart(WheelShowCase.PresetWheelTask);

PicoGK.Library.Go(voxelSize, task, bEndAppWithTask: true);

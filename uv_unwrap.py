import bpy

# Specify the path to your .obj file
obj_path = "/home/steven/itri/tracked_mesh/000220.obj"

# Specify the export path for the UV layout image
export_path = "/home/steven/itri/tracked_mesh/uv_layout2.png"

# Specify the export path for the new .obj file
output_path = "/home/steven/itri/tracked_mesh/000220_new.obj"

# Clear existing objects in the scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Import the .obj file
bpy.ops.import_scene.obj(filepath=obj_path)

# Get the imported mesh object
obj = bpy.context.selected_objects[0] if bpy.context.selected_objects else None

# Check if the object is valid and of type 'MESH'
if not obj or obj.type != 'MESH':
    print("Imported object is not a valid mesh.")
    exit()
    
bpy.context.view_layer.objects.active = obj

# Triangulate the mesh
bpy.ops.object.modifier_add(type='TRIANGULATE')
bpy.ops.object.modifier_apply(modifier="Triangulate")

bpy.ops.object.mode_set(mode='EDIT')

# Select all faces in Edit Mode
bpy.ops.uv.select_all(action='SELECT')

# Create a new UV map for the object
uv_map_name = "UVMap"
obj.data.uv_layers.new(name=uv_map_name)

# Set the active UV map to the newly created UV map
obj.data.uv_layers[uv_map_name].active = True

# Select the newly created UV map
# bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)
bpy.ops.object.mode_set(mode='OBJECT')

# # Export the UV layout
# bpy.ops.uv.export_layout(filepath=export_path, opacity=1.0, size=(1024, 1024), export_all=True)

# print("UV layout exported to:", export_path)

image = bpy.data.images.load(export_path)
width, height = image.size

# Get the UV coordinates for each vertex
uv_coordinates = []
uv_layer = obj.data.uv_layers[uv_map_name]
for loop in obj.data.loops:
    vertex_index = loop.vertex_index
    uv = uv_layer.data[loop.index].uv
    pixel_x = int(uv.x * width)
    pixel_y = int(uv.y * height)

    uv_coordinates.append((vertex_index, pixel_x, pixel_y))

# Print the UV coordinates for each vertex
with open('uv.txt', 'w') as f:
    for vertex_index, pixel_x, pixel_y in uv_coordinates:
        print(f"{vertex_index}, {pixel_x}, {pixel_y}\n")
        f.write(f"{vertex_index}, {pixel_x}, {pixel_y}\n")
        
        
bpy.ops.export_scene.obj(filepath=output_path, use_mesh_modifiers=False,  keep_vertex_order=True)
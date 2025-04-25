from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from plyer import camera

class MyApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        # text = Label(text="Hello World")
        # btn = Button(text="Take Photo", on_press=self.take_picture)
        # layout.add_widget(text)
        # layout.add_widget(btn)
        self.img = Image()
        layout.add_widget(self.img)
        return layout

    def take_picture(self, instance):
        camera.take_picture(filename='photo.jpg', on_complete=self.show_photo)

    def show_photo(self, path):
        # TODO: run model here
        print(f"Photo taken {path}")


if __name__ == '__main__':
    MyApp().run()
